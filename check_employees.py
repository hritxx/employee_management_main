#!/usr/bin/env python3
import pandas as pd
from core.database import get_cursor

# Get CSV names
df = pd.read_csv('allocation/allocations.csv')
csv_names = df['Name'].unique()
print('CSV employee names:')
for name in csv_names:
    print(f'  {repr(name)}')

print('\nDatabase employee names:')
with get_cursor() as cursor:
    cursor.execute('SELECT employee_name FROM employee WHERE status = \'Active\' ORDER BY employee_name')
    db_names = [row[0] for row in cursor.fetchall()]
    for name in db_names:
        print(f'  {repr(name)}')

print('\nComparison:')
for csv_name in csv_names:
    found = csv_name in db_names
    print(f'  {csv_name}: {"Found" if found else "Missing"}')
