import re
import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime

def fetch_sa_lotto_history(output_csv="draws.csv", years_back=5):
    """
    Fetch SA Lotto historical results from lottery.co.za
    Scrapes year archive pages to get historical draw data.
    """
    current_year = datetime.now().year
    all_draws = []
    
    for year in range(current_year, current_year - years_back, -1):
        url = f"https://www.lottery.co.za/lotto/results/{year}"
        try:
            r = requests.get(url, timeout=30)
            if r.status_code != 200:
                print(f"  Skipping {year}: HTTP {r.status_code}")
                continue
                
            soup = BeautifulSoup(r.text, "html.parser")
            rows = soup.select("table tr")
            
            year_count = 0
            for row in rows:
                cells = row.select("td")
                if len(cells) >= 3:
                    # Get date from first cell
                    date_text = cells[0].get_text(strip=True)
                    
                    # Get draw number from second cell
                    draw_id = cells[1].get_text(strip=True)
                    
                    # Get ball numbers from the third cell
                    num_cell = cells[2]
                    balls = num_cell.select(".lotto-ball")
                    
                    # Take only the first 6 balls (main numbers, not bonus)
                    if len(balls) >= 6:
                        nums = [b.get_text(strip=True) for b in balls[:6]]
                        all_draws.append({
                            "draw_id": draw_id,
                            "date": date_text,
                            "numbers": nums
                        })
                        year_count += 1
            
            print(f"  {year}: Found {year_count} draws")
            
        except Exception as e:
            print(f"  Error fetching {year}: {e}")
            continue
    
    # Write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_id", "date", "n1", "n2", "n3", "n4", "n5", "n6"])
        
        for draw in all_draws:
            writer.writerow([
                draw["draw_id"],
                draw["date"]
            ] + draw["numbers"])
    
    print(f"Total: {len(all_draws)} draws saved to {output_csv}")
    return output_csv


def fetch_current_lotto_jackpot(default: float | None = None) -> float | None:
    """
    Scrape the current SA Lotto jackpot from the dedicated Lotto page.
    Returns jackpot in Rands, or default if not found.
    """
    urls = [
        "https://za.national-lottery.com/lotto",
        "https://za.national-lottery.com/lotto/results",
    ]
    
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(" ", strip=True)

            m = re.search(r"Jackpot:\s*R\s*([0-9]+(?:\.[0-9]+)?)\s*Million", text, re.IGNORECASE)
            if m:
                millions = float(m.group(1))
                return millions * 1_000_000.0

            m2 = re.search(r"Jackpot:\s*R\s*([0-9,]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            if m2:
                amt_str = m2.group(1).replace(",", "")
                amt = float(amt_str)
                if amt > 100_000:
                    return amt

        except Exception:
            continue
    
    return default
