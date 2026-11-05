"""Append 13 original NSCG fields to PR17 without cleaning or changing its cohort."""
import csv
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BASE = ROOT / 'notebooks/possible data integration/nscg2023_onet30_2_subset.csv'
RAW = ROOT / 'data/nscg/pcg23Public/epcg23.csv'
DEST = OUT / 'nscg2023_onet30_2_integration63.csv'
ADDED = ['SALARY', 'HRSWK', 'EMSECSM', 'EMSIZE', 'SATADV', 'SATBEN',
         'SATCHAL', 'SATIND', 'SATLOC', 'SATRESP', 'SATSAL', 'SATSEC', 'SATSOC']
EXPECTED = {
    BASE: 'c05e0219ea29738c1ce6ea95aa834c6d2a37aae3b6a90db5d10325b03d9b0f8b',
    RAW: '41b626b26dc63fd0fa4a24d290c3dd5b3bd363a7444b597a17f8d771856d5791',
}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path):
    return path.open(encoding='utf-8-sig', newline='')


def main():
    for path, expected in EXPECTED.items():
        assert sha256(path) == expected, f'Source changed: {path}'

    source = {}
    with read_csv(RAW) as f:
        reader = csv.DictReader(f)
        assert set(ADDED + ['OBSNUM']).issubset(reader.fieldnames)
        for row in reader:
            key = row['OBSNUM'].strip()
            assert key and key not in source
            source[key] = [row[c] for c in ADDED]

    seen = set()
    occupations = Counter()
    partial = DEST.with_suffix('.csv.partial')
    with read_csv(BASE) as f, partial.open('w', encoding='utf-8', newline='') as target:
        reader = csv.reader(f)
        original_columns = next(reader)
        assert len(original_columns) == len(set(original_columns)) == 50
        assert not set(ADDED).intersection(original_columns)
        columns = original_columns + ADDED
        key_position = columns.index('OBSNUM')
        occupation_position = columns.index('N3OCPR')
        writer = csv.writer(target)
        writer.writerow(columns)
        for row in reader:
            assert len(row) == 50
            key = row[key_position].strip()
            assert key and key not in seen and key in source
            seen.add(key)
            occupations[row[occupation_position]] += 1
            writer.writerow(row + source[key])
    assert len(seen) == 74016 and len(occupations) == 123

    distributions = {c: Counter() for c in ADDED}
    checked_rows = 0
    with read_csv(BASE) as base_file, read_csv(partial) as output_file:
        old = csv.DictReader(base_file)
        new = csv.DictReader(output_file)
        assert new.fieldnames == columns
        for previous, current in itertools.zip_longest(old, new):
            assert previous is not None and current is not None
            assert None not in current and all(v is not None for v in current.values())
            assert all(current[c] == previous[c] for c in original_columns)
            assert [current[c] for c in ADDED] == source[previous['OBSNUM'].strip()]
            for c in ADDED:
                distributions[c][current[c]] += 1
            checked_rows += 1
    assert checked_rows == 74016

    checks = {}
    for c in ADDED:
        values = distributions[c]
        assert all(v == v.strip() and v != '' for v in values)
        if c.startswith('SAT'):
            assert set(values) <= {'1', '2', '3', '4'}
        elif c == 'EMSECSM':
            assert set(values) <= {'1', '2', '3'}
        elif c == 'EMSIZE':
            assert set(values) <= set('12345678')
        elif c == 'SALARY':
            assert all(v.isdigit() and int(v) != 9999998 for v in values)
        elif c == 'HRSWK':
            assert all(v.isdigit() and 1 <= int(v) <= 96 for v in values)
        checks[c] = {'blank_count': 0, 'valid_code_check': 'passed'}
        if c in ('SALARY', 'HRSWK'):
            checks[c].update(minimum=min(map(int, values)), maximum=max(map(int, values)))
        else:
            checks[c]['value_counts'] = dict(sorted(values.items()))

    for path, expected in EXPECTED.items():
        assert sha256(path) == expected
    if DEST.exists():
        assert sha256(DEST) == sha256(partial), 'Existing output differs; refusing overwrite'
        partial.unlink()
    else:
        partial.replace(DEST)

    report = {
        'status': 'passed', 'respondents': checked_rows, 'columns': len(columns),
        'occupation_categories': len(occupations), 'raw_source_rows': len(source),
        'added_columns': ADDED, 'duplicate_ids': 0, 'unmatched_ids': 0,
        'original_50_columns_and_row_order_unchanged': True,
        'added_values_equal_raw_source': True, 'source_files_unchanged': True,
        'original_values_checked': checked_rows * 50,
        'added_values_checked': checked_rows * len(ADDED),
        'source_sha256': {str(p.relative_to(ROOT)): h for p, h in EXPECTED.items()},
        'output_file': DEST.name, 'output_sha256': sha256(DEST),
        'field_checks': checks,
        'policy': 'Append only these 13 fields by unique OBSNUM. Preserve original CSV field values. No recoding, reverse coding, imputation, row filtering, outlier removal, scaling, or model training. Key whitespace is stripped only for matching.',
    }
    (OUT / 'integration63_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'integration63_README.txt').write_text(
        'PR17 integration with 13 additional NSCG fields\n\n'
        '74,016 respondents; 123 occupation categories; 63 columns.\n'
        'The first 50 columns and row order are identical to the PR17 input.\n'
        'The last 13 columns retain original NSCG values, joined by OBSNUM:\n'
        + ', '.join(ADDED) + '\n\n'
        'SAT fields: 1 = very satisfied, 2 = somewhat satisfied,\n'
        '3 = somewhat dissatisfied, 4 = very dissatisfied.\n'
        'EMSECSM and EMSIZE are category codes, not measured numeric amounts.\n'
        'HRSWK value 96 represents 96 hours or more.\n'
        'No cleaning or model training was performed. Validation checks do not alter data.\n'
        'Variable definitions and codes: data/nscg/pcg23Public/Dpcg23.xlsx and Ppcg23.sas.\n'
        'Reproduce: python build_integration63.py\n'
        'Verification: integration63_verification.json. Original inputs remain unchanged.\n'
    )
    print(json.dumps({k: report[k] for k in ['status', 'respondents', 'columns',
        'occupation_categories', 'original_values_checked', 'added_values_checked',
        'output_sha256']}, indent=2))


if __name__ == '__main__':
    main()
