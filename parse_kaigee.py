import json
import csv
import sys

input_file = r"C:\Users\manue\.gemini\antigravity\brain\b4921ab0-19c0-45e7-b74f-bbcf5bdc76fb\.system_generated\steps\245\content.md"
output_file = "kaigee_history.csv"

def convert():
    print(f"Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        # The file has frontmatter from the download tool, so we need to skip lines until the first '['
        lines = f.readlines()
        
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('['):
            start_idx = i
            break
            
    json_str = "".join(lines[start_idx:])
    data = json.loads(json_str)
    
    print(f"Loaded {len(data)} records. Writing to {output_file}...")
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(['w1', 'w2', 'w3', 'w4', 'w5', 'm1', 'm2', 'm3', 'm4', 'm5', 'date', 'event', 'ev_id'])
        
        for row in data:
            # Kaigee keys:
            # Fbw, Sbw, Cbw, Ftbw, Lbw (Winning numbers)
            # Fbm, Sbm, Cbm, Ftbm, Lbm (Machine numbers)
            w1 = row.get('Fbw', '')
            w2 = row.get('Sbw', '')
            w3 = row.get('Cbw', '')
            w4 = row.get('Ftbw', '')
            w5 = row.get('Lbw', '')
            
            m1 = row.get('Fbm', '')
            m2 = row.get('Sbm', '')
            m3 = row.get('Cbm', '')
            m4 = row.get('Ftbm', '')
            m5 = row.get('Lbm', '')
            
            date = row.get('Date', '')
            event = row.get('Game', '')
            ev_id = row.get('Evt', '')
            
            writer.writerow([w1, w2, w3, w4, w5, m1, m2, m3, m4, m5, date, event, ev_id])
            
    print("Done!")

if __name__ == "__main__":
    convert()
