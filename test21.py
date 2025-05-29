# Program to find keywords and associated dollar amounts in omnibus bills
from bs4 import BeautifulSoup, UnicodeDammit  # pip install bs4
import re
import numpy as np
import csv
from datetime import datetime
import os
from collections import defaultdict
import html  # Add this at the top if not already present
from sklearn.metrics.pairwise import cosine_similarity
import hashlib

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
from datetime import datetime

# Global constants
bill_filenames = [
    'HR2617_2022-12-29.txt',
    'HR4366_2024-03-09.txt',
    'HR2471_2022-03-15.txt',
    'HR133_2020-12-27.txt',
    'HR1865_2019-12-20.txt',
    'HR6157_2018-09-28.txt',
    'HR1158_2019-12-20.txt'
]

# Big Beautiful Bill (HR1)
#bill_filenames = ['HR1_2025-05-20.txt']

# Enhanced regex for dollar amounts with optional billion/million/trillion suffixes
dollar_regex = re.compile(
    r'\$\s?\d[\d,]*(?:\.\d{2})?\s*(?:billion|million|trillion)?',
    re.IGNORECASE
)

results = []
no_dollar_results = []
all_semantic_results = []

# Functions
def extract_date_from_filename(filename):
    basename = os.path.basename(filename)
    match = re.search(r'\d{4}-\d{2}-\d{2}', basename)
    return match.group() if match else "Unknown"

def parse_dollar_amounts(matches):
    parsed = []
    seen = set()
    for match in matches:
        cleaned = match.lower().replace("$", "").replace(",", "").strip()
        multiplier = 1
        if "billion" in cleaned:
            multiplier = 1e9
            cleaned = cleaned.replace("billion", "").strip()
        elif "million" in cleaned:
            multiplier = 1e6
            cleaned = cleaned.replace("million", "").strip()
        elif "trillion" in cleaned:
            multiplier = 1e12
            cleaned = cleaned.replace("trillion", "").strip()
        try:
            amount = float(cleaned) * multiplier
            if amount not in seen:
                seen.add(amount)
                parsed.append(amount)
        except ValueError:
            continue
    return parsed

def clean_heading(text):
    if not text:
        return "Unknown"
    # Unescape HTML entities (e.g., "&lt;" to "<")
    text = html.unescape(text)
    # Remove stray angle brackets or tags
    text = re.sub(r'[<>]', '', text)
    # Collapse multiple spaces and line breaks
    text = " ".join(text.split())
    return text.strip()

def extract_divisions_and_titles(text):
    divisions_titles = []

    division_pattern = re.compile(r"(DIVISION\s+[A-Z]+(?:[\s\-–—:]+[^\n]+)?)", re.IGNORECASE)
    title_pattern = re.compile(r"(TITLE\s+[IVXLC]+(?:[\s\-–—:]+[^\n]+)?)", re.IGNORECASE)

    current_division = None
    for match in re.finditer(rf"{division_pattern.pattern}|{title_pattern.pattern}", text):
        heading = clean_heading(match.group().strip())
        start_pos = match.start()

        if heading.upper().startswith("DIVISION"):
            current_division = heading
            divisions_titles.append((start_pos, heading, None))
        elif heading.upper().startswith("TITLE") and current_division:
            divisions_titles.append((start_pos, current_division, heading))

    return sorted(divisions_titles, key=lambda x: x[0])

def load_glove_embeddings(glove_file_path):
    embeddings = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    return embeddings

def get_phrase_vector(phrase, embeddings):
    words = phrase.lower().split()
    vectors = [embeddings[word] for word in words if word in embeddings]
    if vectors:
        return np.mean(vectors, axis=0)
    return None

def find_semantic_matches(text, keyword_vector, embeddings, window_size=20, threshold=0.7):
    words = text.split()
    matches = []

    for i in range(len(words) - window_size + 1):
        window_words = words[i:i+window_size]
        window_vectors = [embeddings[word] for word in window_words if word in embeddings]
        
        if window_vectors:
            window_vector = np.mean(window_vectors, axis=0).reshape(1, -1)
            sim = cosine_similarity(keyword_vector.reshape(1, -1), window_vector)[0][0]

            if sim >= threshold:
                matched_text = " ".join(window_words)
                matches.append((i, matched_text, sim))
    
    return matches

def get_paragraph(text, idx, window=500):
    """
    Grab a full paragraph (bounded by newlines or periods) surrounding index `idx`.
    Also return the start index of the paragraph in the full text.
    """
    start = text.rfind('.', 0, idx)
    end = text.find('.', idx)
    if start == -1: start = max(0, idx - window)
    if end == -1: end = min(len(text), idx + window)
    
    para_text = text[start+1:end+1].strip()
    para_start = start + 1  # +1 to skip the period
    
    return para_text, para_start

def get_top_related_words(keyword_vector, embeddings, top_n=200):
    similarities = []
    for word, vec in embeddings.items():
        sim = np.dot(vec, keyword_vector) / (np.linalg.norm(vec) * np.linalg.norm(keyword_vector))
        similarities.append((word, sim))

    # Sort and return top N
    top_words = sorted(similarities, key=lambda x: x[1], reverse=True)[:top_n]
    return [word for word, _ in top_words]

def highlight_dollar_amounts(text):
    """Bold green highlight for dollar amounts like $1,000,000"""
    return re.sub(r'(\$\s?\d[\d,]*(?:\.\d{2})?)', r'\033[1;32m\1\033[0m', text)

def highlight_related_words(text, related_words):
    """Bold cyan highlight for top related words"""
    for word in related_words:
        pattern = re.compile(rf'\b({re.escape(word)})\b', re.IGNORECASE)
        text = pattern.sub(r'\033[1;36m\1\033[0m', text)
    return text

# Semantic search code
def run_semantic_search(keyword, embeddings, bill_filenames, similarity_threshold):
    keyword_vector = get_phrase_vector(keyword, embeddings)
    if keyword_vector is None:
        print(f"⚠️ Keyword '{keyword}' not found in GloVe vocabulary. Skipping semantic search.")
        return []

    top_related_words = get_top_related_words(keyword_vector, embeddings, top_n=40)
    semantic_results = []

    for bill_file in bill_filenames:
        with open(bill_file, encoding='utf-8') as hand:
            soup = BeautifulSoup(hand, 'html.parser')
            full_text = soup.get_text(" ", strip=True)

        div_title_spans = extract_divisions_and_titles(full_text)
        bill_date = extract_date_from_filename(bill_file)
        semantic_matches = find_semantic_matches(full_text, keyword_vector, embeddings, window_size=25, threshold=0.7)
        semantic_matches.sort(key=lambda x: x[2], reverse=True)

        seen_paragraph_hashes = set()

        for idx, context, similarity in semantic_matches:
            if similarity < similarity_threshold:
                continue

            paragraph, para_start = get_paragraph(full_text, idx)
            paragraph_key = hashlib.md5(" ".join(paragraph.lower().split()).encode()).hexdigest()

            dollar_matches = dollar_regex.findall(paragraph)
            parsed_amounts = parse_dollar_amounts(dollar_matches)
            total = sum(parsed_amounts) if parsed_amounts else 0

            if paragraph_key in seen_paragraph_hashes or not parsed_amounts:
                continue
            seen_paragraph_hashes.add(paragraph_key)

            matched_div, matched_title = "Unknown", "Unknown"
            for i in range(len(div_title_spans)):
                if div_title_spans[i][0] > para_start:
                    break
                matched_div, matched_title = div_title_spans[i][1], div_title_spans[i][2] or "Unknown"

            semantic_results.append([
                bill_date,
                bill_file,
                matched_div,
                matched_title,
                paragraph.strip(),
                *parsed_amounts,
                total,
                similarity,
                keyword  # ⬅️ Add this
            ])

            # Print context (optional)
            highlighted = highlight_dollar_amounts(highlight_related_words(paragraph, top_related_words))
            print(f"\n[Semantic Match] {bill_date} | {bill_file} | sim: {similarity:.2f}")
            print(f"📚 Division: {matched_div.strip()} | 📄 Title: {matched_title.strip()}")
            print("..." + highlighted + "...")
            print(f"💵 Parsed: \033[1;32m{parsed_amounts}\033[0m | Total: \033[1;32m{total:,.2f}\033[0m")

    return semantic_results

def export_semantic_results(all_semantic_results, filename="all_semantic_matches.csv"):
    if not all_semantic_results:
        print("\n⚠️ No semantic matches to export.")
        return

    # Determine max number of dollar columns for consistent formatting
    max_semantic_dollars = max(len(row[5:-3]) for row in all_semantic_results)

    # Define headers
    headers = (
        ["Keyword", "Bill Date", "Bill Filename", "Division", "Title", "Context"]
        + [f"Dollar {i+1}" for i in range(max_semantic_dollars)]
        + ["Total", "Similarity Score"]
    )

    # Write to CSV
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in all_semantic_results:
            keyword = row[-1]
            main_part = row[:5]
            dollars = row[5:-3]
            suffix = row[-3:-1]  # Total and Similarity Score

            num_dollars = len(dollars)
            padded_row = [keyword] + main_part + dollars + [""] * (max_semantic_dollars - num_dollars) + suffix
            writer.writerow(padded_row)

    print(f"\n✅ Exported {len(all_semantic_results)} total semantic matches to {filename}")

def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0

def plot_line_chart_by_year(bill_totals):
    """
    Creates a time-series line plot of dollar totals by year, grouped by keyword.
    Shaded backgrounds represent presidential terms.
    
    Parameters:
        bill_totals (dict): {(keyword, date_str, filename): total_amount}
    """
    # Step 1: Aggregate totals by keyword and year
    keyword_year_totals = defaultdict(lambda: defaultdict(float))  # {keyword: {year: total}}

    for (keyword, date_str, filename), total in bill_totals.items():
        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
            keyword_year_totals[keyword][year] += total
        except ValueError:
            print(f"⚠️ Skipping invalid date: {date_str}")

    # Step 2: Prepare plot
    fig, ax = plt.subplots(figsize=(14, 7))

    all_years = set()
    for keyword, year_data in keyword_year_totals.items():
        sorted_years = sorted(year_data)
        totals = [year_data[year] for year in sorted_years]
        ax.plot(sorted_years, totals, marker='o', label=keyword)
        all_years.update(sorted_years)

    # Step 3: Format x-axis with consistent spacing
    if all_years:
        all_years = sorted(all_years)
        ax.set_xticks(all_years)
        ax.set_xlim(min(all_years) - 1, max(all_years) + 1)

    ax.set_xlabel("Year")
    ax.set_ylabel("Total Matched Dollar Amount")
    ax.set_title("Yearly Matched Dollar Totals by Keyword")
    ax.legend(title="Keyword")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Step 4: Add presidential background shading
    presidents = [
        ("Obama", 2009, 2017, "#add8e6"),  # Light blue
        ("Trump", 2017, 2021, "#f4cccc"),  # Light red
        ("Biden", 2021, 2025, "#add8e6"),  # Light blue
    ]
    for _, start_year, end_year, color in presidents:
        ax.axvspan(start_year, end_year, color=color, alpha=0.2)

    plt.tight_layout()
    plt.show()
    
def plot_bar_chart_by_bill(bill_totals):
    """
    Plots a grouped bar chart of total matched dollar amounts by year and keyword.
    Bars are centered correctly even with a single keyword.
    """

    from collections import defaultdict
    from datetime import datetime
    import matplotlib.pyplot as plt

    # Step 1: Group totals by keyword and year
    keyword_year_totals = defaultdict(lambda: defaultdict(float))  # {keyword: {year: total}}

    for (keyword, date_str, filename), total in bill_totals.items():
        try:
            year = datetime.strptime(date_str, "%Y-%m-%d").year
        except ValueError:
            print(f"⚠️ Skipping invalid date: {date_str}")
            continue
        keyword_year_totals[keyword][year] += total

    if not keyword_year_totals:
        print("⚠️ No data to plot.")
        return

    all_years = sorted({year for data in keyword_year_totals.values() for year in data})
    year_to_index = {year: idx for idx, year in enumerate(all_years)}
    num_keywords = len(keyword_year_totals)
    bar_width = 0.8 / max(1, num_keywords)

    plt.figure(figsize=(14, 6))
    colors = plt.cm.tab10.colors

    # Step 2: Plot bars
    for i, (keyword, year_data) in enumerate(keyword_year_totals.items()):
        if num_keywords == 1:
            # Center bars directly on year index
            x_vals = [year_to_index[year] for year in year_data]
        else:
            # Offset each keyword's bars
            x_vals = [year_to_index[year] + i * bar_width for year in year_data]

        y_vals = [year_data[year] for year in year_data]
        plt.bar(x_vals, y_vals, width=bar_width, label=keyword, color=colors[i % len(colors)])

    # Step 3: Format x-axis
    center_offsets = 0 if num_keywords == 1 else (bar_width * (num_keywords - 1)) / 2
    tick_positions = [i + center_offsets for i in range(len(all_years))]

    plt.xticks(ticks=tick_positions, labels=[str(year) for year in all_years])
    plt.xlabel("Year")
    plt.ylabel("Total Matched Dollar Amount")
    plt.title("Yearly Matched Dollar Totals by Keyword")
    if num_keywords > 1:
        plt.legend(title="Keyword")
    plt.tight_layout()
    plt.savefig("grouped_bar_by_keyword.png", dpi=300)
    plt.show()

# Main execution block
while True:
    prev_result_len = len(results)
    
    keyword = input("\n🔍 Enter a keyword for search (or press Enter to quit): ").strip()
    if not keyword:
        print("👋 Exiting keyword search loop.")
        break
    
    # Glove model basics for related-matches
    glove_path = "glove.6B.100d.txt"  # Download from https://nlp.stanford.edu/projects/glove/
    embeddings = load_glove_embeddings(glove_path)
    
    seen_contexts = set()
    
    for bill_file in bill_filenames:
        print(f"\n--- Analyzing {bill_file} ---")

        seen_dollars_in_bill = set()  # Resets for each bill
        with open(bill_file, encoding='utf-8') as hand:
            soup = BeautifulSoup(hand, 'html.parser')
            full_text = soup.get_text(" ", strip=True)

        div_title_spans = extract_divisions_and_titles(full_text)
        bill_date = extract_date_from_filename(bill_file)

        highlight_color = "\033[93m"  # Yellow for keyword
        dollar_color = "\033[1;92m"   # Bold green for dollar amounts
        reset_color = "\033[0m"

        pattern = rf"\b{re.escape(keyword)}\b"
        for m in re.finditer(pattern, full_text, re.IGNORECASE):

            context, _ = get_paragraph(full_text, m.start())

            normalized = " ".join(context.lower().split())
            if normalized in seen_contexts:
                continue
            seen_contexts.add(normalized)

            dollar_matches = dollar_regex.findall(context)
            raw_amounts = parse_dollar_amounts(dollar_matches)

            parsed_amounts = []
            for amt in raw_amounts:
                if amt not in seen_dollars_in_bill:
                    seen_dollars_in_bill.add(amt)
                    parsed_amounts.append(amt)

            if parsed_amounts:
                total = sum(parsed_amounts)

                # Find the last Division/Title before this context
                matched_div = matched_title = "Unknown"
                for i in range(len(div_title_spans)):
                    if div_title_spans[i][0] > m.start():
                        break
                    matched_div, matched_title = div_title_spans[i][1], div_title_spans[i][2]

                matched_div = clean_heading(matched_div)
                matched_title = clean_heading(matched_title)
        
                results.append([
                    bill_date,
                    bill_file,
                    matched_div,
                    matched_title,
                    context.strip(),
                    *parsed_amounts,
                    total,
                    keyword  # ⬅️ Add this
                ])
    
                print(f"\n[Matched] {bill_date} | {bill_file}")
                print(f"📚 Division: {matched_div}")
                print(f"📄 Title: {matched_title}")

                # Highlight keyword in context
                highlighted_context = re.sub(
                    rf"({re.escape(keyword)})",
                    rf"{highlight_color}\1{reset_color}",
                    context,
                    flags=re.IGNORECASE
                )

                # Highlight dollar amounts
                highlighted_context = re.sub(
                    dollar_regex,
                    lambda m: f"{dollar_color}{m.group()}{reset_color}",
                    highlighted_context
                )
                print("..." + highlighted_context + "...")

                print(f"💵 Parsed Amounts: {parsed_amounts} | Total: {total:,.2f}")
            else:
                # Find the last Division/Title before this context
                matched_div = matched_title = "Unknown"
                for i in range(len(div_title_spans)):
                    if div_title_spans[i][0] > m.start():
                        break
                    matched_div, matched_title = div_title_spans[i][1], div_title_spans[i][2]

                matched_div = clean_heading(matched_div)
                matched_title = clean_heading(matched_title)

                # Save matches with no dollar amounts
                no_dollar_results.append([
                    keyword, bill_date, bill_file, matched_div, matched_title, context, 'NO_DOLLARS_FOUND'
                ])
    
                 # Highlight keyword in context
                highlighted_context = re.sub(
                    rf"({re.escape(keyword)})",
                    rf"{highlight_color}\1{reset_color}",
                    context,
                    flags=re.IGNORECASE
                )

                print(f"\n⚠️ Keyword match with NO dollar amounts")
                print(f"📅 {bill_date} | 📄 {bill_file}")
                print(f"📚 Division: {matched_div}\n📄 Title: {matched_title}")
                print("..." + highlighted_context + "...")
                
        # ---- Keyword Search Summary ----
        # After keyword search completes
        current_results = results[prev_result_len:]  # Only new results from this keyword
        num_matches = len(current_results)
        total_amount = sum(row[-2] for row in current_results) if current_results else 0

        print(f"\n🔎 Keyword Search Summary for: '{keyword}'")
        print(f"📄 Matched Contexts with Dollar Amounts: {num_matches}")
        print(f"💰 Total Matched Dollar Amount: ${total_amount:,.2f}")
    
    run_semantic_input = input(f"\n🤖 Would you like to find related (semantic) matches for '{keyword}' and export dollar amounts? (y/n): ").strip().lower()
    run_semantic = run_semantic_input == 'y'

    if run_semantic:
        similarity_threshold = float(input(
            "Set minimum similarity score for semantic matches (e.g., 0.75).\n"
            "Scores range from 0 to 1 — higher scores mean fewer, but stronger, matches.\n"
            "✔️ 0.70+ = good | ✔️ 0.80+ = excellent\n"
            "Your threshold (default 0.75): "
        ) or "0.75")

        print(f"\n🔍 Running semantic search for related contexts using GloVe for '{keyword}'...")
        semantic_results = run_semantic_search(keyword, embeddings, bill_filenames, similarity_threshold)
        all_semantic_results.extend(semantic_results)
 
# Export to .csv: Bill Date, Bill Filename, Dollars, Total
# Determine max number of dollar columns for consistent CSV output
if results:
    max_dollars = max(len(row[5:-2]) for row in results)

    with open("all_keywords_dollar_contexts_with_totals.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        headers = (
            ["Keyword", "Bill Date", "Bill Filename", "Division", "Title", "Context"]
            + [f"Dollar {i+1}" for i in range(max_dollars)]
            + ["Total"]
        )
        writer.writerow(headers)

        for row in results:
            keyword = row[-1]             # ✅ Now at the end
            metadata = row[0:5]           # ✅ Bill Date to Context
            dollar_values = row[5:-2]     # ✅ All $ values except Total
            total = row[-2]               # ✅ Total right before keyword

            padded_dollars = dollar_values + [""] * (max_dollars - len(dollar_values))

            writer.writerow([keyword] + metadata + padded_dollars + [total])

    print(f"\n✅ Exported {len(results)} rows to {"all_keywords_dollar_contexts_with_totals.csv"}")

    # Totals Export: Bill Date, Bill Filename, and Total (dollar totals per bill)
    bill_totals = defaultdict(float)

    for row in results:
        keyword = row[-1]
        date = row[0]
        filename = row[1]
        division = row[2]
        total = safe_float(row[-2])

        bill_key = (keyword, date, filename)
        bill_totals[bill_key] += total

    summary_csv = "all_keywords_totals_by_bill.csv"

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Keyword", "Bill Date", "Bill Filename", "Total Matched Amount"])
        for (keyword, date, filename), total in sorted(bill_totals.items()):
            writer.writerow([keyword, date, filename, total])

    print(f"\n📊 Exported {len(bill_totals)} rows to {summary_csv}")

    # Prepare data for plotting (total dollar amounts per bill)
    plot_choice = input("\n📊 Would you like a plot? Enter 'bar' for bar chart, 'line' for time-series line plot, or 'none' to skip: ").strip().lower()

    if plot_choice == 'bar':
        # >>> Call your bar plot function here (based on bill_totals)
        plot_bar_chart_by_bill(bill_totals)
    elif plot_choice == 'line':
        # >>> Call your line plot function here (based on keyword_year_totals)
        plot_line_chart_by_year(bill_totals)
    elif plot_choice == 'none':
        print("🛑 Skipping plot generation.")
    else:
        print("⚠️ Invalid option. Please enter 'bar', 'line', or 'none'.")

# No dollar matches export
if no_dollar_results:
    # export keyword matches that had no dollar amounts
    no_matches_file = "no_dollar_matches_for_keyword.csv"
    with open(no_matches_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Keyword", "Bill Date", "Bill Filename", "Division", "Title", "Context", "Note"])
        for row in no_dollar_results:
            writer.writerow(row)  # No need to prepend keyword anymore

    print(f"\n📄 Exported {len(no_dollar_results)} matches without dollar amounts to {no_matches_file}")

export_semantic_results(all_semantic_results)