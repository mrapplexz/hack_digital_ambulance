import sys
from pathlib import Path
from typing import Iterator, Any, List, Dict
import datetime as dt

import pandas as pd
import pyexcel as pex
from tqdm import tqdm


def parse_sheet(sheet: List[List[Any]]) -> List[Dict[str, Any]]:
    sheet_bar = tqdm(sheet)
    sheet_iter = iter(sheet_bar)

    report_date = next(sheet_iter)  # skip
    if next(sheet_iter)[0] != 'Журнал Активных вызовов':
        raise ValueError('Invalid sheet format :/')
    next(sheet_iter)  # skip

    blocks = []
    try:
        skipped_rows = 0
        while True:
            sheet_bar.refresh()
            try:
                block = parse_block(sheet_iter)
                blocks.append(block)
            except ValueError:
                skipped_rows += 1
                sheet_bar.set_postfix({'skipped_rows': skipped_rows})
                sheet_bar.refresh()
    except StopIteration:
        pass
    sheet_bar.refresh()
    return blocks


def parse_file(file: Path):
    print(f':: Parsing file {file}')
    book = pex.get_book_dict(file_name=str(file))
    blocks = []
    for sheet_name, sheet in book.items():
        blocks.extend(parse_sheet(sheet))
    return blocks


def raise_for_prefill(value: Any, prefill_text: str):
    if not isinstance(value, str):
        raise ValueError('Prefill is not a string!')
    if value.strip() != prefill_text.capitalize():
        raise ValueError('Invalid prefill text!')


def parse_block(sheet_iter: Iterator[List[Any]]):
    out_dict = {}

    row1 = next(sheet_iter)
    raise_for_prefill(row1[2], 'Номер:')
    raise_for_prefill(row1[4], 'Больной:')
    raise_for_prefill(row1[14], 'Возраст:')
    out_dict['call_date'] = str(row1[0])
    out_dict['call_number'] = str(row1[3])
    out_dict['patient_age'] = str(row1[16])
    out_dict['call_initiator'] = str(row1[19])

    row2 = next(sheet_iter)
    raise_for_prefill(row2[0], 'Адрес:')
    out_dict['patient_address'] = str(row2[1])

    row3 = next(sheet_iter)
    raise_for_prefill(row3[0], 'Повод:')
    raise_for_prefill(row3[9], 'Вызов:')
    raise_for_prefill(row3[15], 'Вид:')
    out_dict['call_reason'] = str(row3[1])
    out_dict['call_order'] = str(row3[11])
    out_dict['call_type'] = str(row3[18])

    row4 = next(sheet_iter)
    raise_for_prefill(row4[0], 'Диагноз:')
    raise_for_prefill(row4[9], 'Результат:')
    out_dict['patient_diagnosis'] = str(row4[1])
    out_dict['call_result'] = str(row4[11])

    row5 = next(sheet_iter)
    raise_for_prefill(row5[0], 'Доставлен:')
    raise_for_prefill(row5[7], 'Бригада:')
    raise_for_prefill(row5[12], 'Подстанция:')
    out_dict['hospitalized_to'] = str(row5[1])
    out_dict['substation'] = str(row5[17])

    row6 = next(sheet_iter)
    raise_for_prefill(row6[0], 'Принят:')
    raise_for_prefill(row6[10], 'Приезд:')
    raise_for_prefill(row6[13], 'Госпит-ан:')
    raise_for_prefill(row6[19], 'Испол.')
    call_time = row6[8]
    out_dict['call_time'] = call_time.strftime('%H:%M:%S') if isinstance(call_time, dt.datetime) else call_time
    arrival_time = row6[11]
    out_dict['arrival_time'] = arrival_time.strftime('%H:%M:%S') if isinstance(arrival_time, dt.datetime) else \
        arrival_time

    return out_dict


if __name__ == '__main__':

    print(': Parsing files')
    files = list(Path('data').rglob('*.xls'))
    blocks = []
    for file in files:
        blocks.extend(parse_file(file))

    print(': Writing .csv')
    blocks_df = pd.DataFrame(blocks)
    blocks_df.to_csv('data_processed.csv', index=False)
