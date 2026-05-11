# pip install pymupdf pandas

import fitz
import re
import pandas as pd


PDF_PATH = "ปีการศึกษา 2568 - หลักสูตรปริญญาตรี คณะวิศวกรรมศาสตร์.pdf"


def clean_line(line):
    line = line.replace("\u00a0", " ")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def normalize_course_id(text):
    return re.sub(r"\s+", " ", text).strip()


def extract_lines_from_pdf(pdf_path, start_page=67):
    doc = fitz.open(pdf_path)
    lines = []

    for page_index in range(start_page - 1, len(doc)):
        page_text = doc[page_index].get_text()
        for line in page_text.split("\n"):
            line = clean_line(line)
            if line:
                lines.append(line)

    doc.close()
    return lines


def is_header_footer(line):
    if "หลักสูตรปริญญาตรี คณะวิศวกรรมศาสตร์ ปีการศึกษา 2568" in line:
        return True
    if re.fullmatch(r"\d+", line):
        return True
    return False


def parse_courses(lines):
    thai_code_pattern = re.compile(r"^[ก-ฮ]{1,4}\.\s*\d{3}$")
    eng_code_pattern = re.compile(
        r"^(EN|GE|MA|PH|CH|CS|EE|EL|IE|ME|CE|MI|AIE|CO)\s*\d{3}$"
    )
    credit_pattern = re.compile(
        r"^\d+\s*\(\s*\d+\s*[–-]\s*\d+\s*[–-]\s*\d+\s*\)$"
    )

    lines = [line for line in lines if not is_header_footer(line)]

    rows = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not thai_code_pattern.match(line):
            i += 1
            continue

        # Find next course boundary
        j = i + 1
        while j < len(lines) and not thai_code_pattern.match(lines[j]):
            j += 1

        block = lines[i:j]

        course_id = None
        subject = None
        content = []

        # Subject = first useful Thai line after Thai code
        for item in block[1:]:
            if credit_pattern.match(item):
                continue
            if eng_code_pattern.match(item):
                continue
            if re.search(r"[A-Za-z]{3,}", item):
                continue
            if item.startswith(("คำอธิบายรายวิชา", "สาขาวิชา", "หมวดวิชา", "กลุ่มวิชา")):
                continue

            subject = item
            break

        # Find English course ID
        eng_idx = None
        credit_idx = None

        for idx, item in enumerate(block):
            if eng_code_pattern.match(item):
                course_id = normalize_course_id(item)
                eng_idx = idx

            if credit_pattern.match(item):
                credit_idx = idx

        if not course_id or not subject:
            i = j
            continue

        # Content starts after:
        # - credit line
        # - English ID line
        # - English course name line
        start_idx_candidates = []

        if credit_idx is not None:
            start_idx_candidates.append(credit_idx + 1)

        if eng_idx is not None:
            start_idx_candidates.append(eng_idx + 2)

        start_idx = max(start_idx_candidates) if start_idx_candidates else 2

        for item in block[start_idx:]:
            if eng_code_pattern.match(item):
                continue
            if credit_pattern.match(item):
                continue

            # Stop when English description starts
            if re.search(r"[A-Za-z]{4,}", item):
                break

            if item.startswith(("คำอธิบายรายวิชา", "สาขาวิชา", "หมวดวิชา", "กลุ่มวิชา")):
                continue

            content.append(item)

        if content:
            rows.append({
                "ID": course_id,
                "subject": subject,
                "content": " ".join(content)
            })

        i = j

    return pd.DataFrame(rows)


if __name__ == "__main__":
    lines = extract_lines_from_pdf(PDF_PATH, start_page=67)

    df = parse_courses(lines)

    print(df.head())
    print("Total rows:", len(df))

    df.to_csv("rag_course_descriptions.csv", index=False, encoding="utf-8-sig")
    df.to_json(
        "rag_course_descriptions.json",
        orient="records",
        force_ascii=False,
        indent=2
    )

    print("Saved rag_course_descriptions.csv")
    print("Saved rag_course_descriptions.json")

    print(df[df["ID"] == "EN 101"])