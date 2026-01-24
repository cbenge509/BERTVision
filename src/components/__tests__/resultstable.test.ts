/**
 * Unit tests for ResultsTable component functionality.
 */

import {describe, it, expect, beforeEach, afterEach} from 'vitest';
import {JSDOM} from 'jsdom';

describe('ResultsTable', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="results-table-wrapper">
            <table class="results-table" role="grid">
              <thead>
                <tr>
                  <th scope="col" class="metric-header" data-sort="metric">
                    <button type="button" class="sort-button" aria-label="Sort by Metric">
                      Metric
                      <span class="sort-indicator" aria-hidden="true"></span>
                    </button>
                  </th>
                  <th scope="col" data-sort="bertBase">
                    <button type="button" class="sort-button" aria-label="Sort by BERT-base">
                      BERT-base
                      <span class="sort-indicator" aria-hidden="true"></span>
                    </button>
                  </th>
                  <th scope="col" data-sort="bertLarge">
                    <button type="button" class="sort-button" aria-label="Sort by BERT-large">
                      BERT-large
                      <span class="sort-indicator" aria-hidden="true"></span>
                    </button>
                  </th>
                  <th scope="col" class="bv-header" data-sort="bvBase">
                    <button type="button" class="sort-button" aria-label="Sort by BERTVision-base">
                      BV-base
                      <span class="sort-indicator" aria-hidden="true"></span>
                    </button>
                  </th>
                  <th scope="col" class="bv-header" data-sort="bvLarge">
                    <button type="button" class="sort-button" aria-label="Sort by BERTVision-large">
                      BV-large
                      <span class="sort-indicator" aria-hidden="true"></span>
                    </button>
                  </th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td class="metric-name">Accuracy</td>
                  <td class="metric-value" data-column="bertBase" data-value="0.85">85.00%</td>
                  <td class="metric-value" data-column="bertLarge" data-value="0.88">88.00%</td>
                  <td class="metric-value bv-value best-value" data-column="bvBase" data-value="0.90">90.00%</td>
                  <td class="metric-value bv-value" data-column="bvLarge" data-value="0.89">89.00%</td>
                </tr>
                <tr>
                  <td class="metric-name">F1</td>
                  <td class="metric-value" data-column="bertBase" data-value="0.82">82.00%</td>
                  <td class="metric-value" data-column="bertLarge" data-value="0.84">84.00%</td>
                  <td class="metric-value bv-value" data-column="bvBase" data-value="0.86">86.00%</td>
                  <td class="metric-value bv-value best-value" data-column="bvLarge" data-value="0.87">87.00%</td>
                </tr>
                <tr>
                  <td class="metric-name">Precision</td>
                  <td class="metric-value best-value" data-column="bertBase" data-value="0.92">92.00%</td>
                  <td class="metric-value" data-column="bertLarge" data-value="0.90">90.00%</td>
                  <td class="metric-value bv-value" data-column="bvBase" data-value="0.88">88.00%</td>
                  <td class="metric-value bv-value" data-column="bvLarge" data-value="0.89">89.00%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  afterEach(() => {
    dom.window.close();
  });

  describe('Table structure', () => {
    it('should have proper table semantics', () => {
      const table = document.querySelector('.results-table');
      expect(table).not.toBeNull();
      expect(table?.getAttribute('role')).toBe('grid');
    });

    it('should have sortable column headers', () => {
      const sortHeaders = document.querySelectorAll('th[data-sort]');
      expect(sortHeaders.length).toBe(5);
    });

    it('should have sort buttons in headers', () => {
      const sortButtons = document.querySelectorAll('.sort-button');
      expect(sortButtons.length).toBe(5);
    });

    it('should mark best values in each row', () => {
      const bestValues = document.querySelectorAll('.best-value');
      expect(bestValues.length).toBe(3); // One per row
    });
  });

  describe('Sorting logic', () => {
    function initSorting(): void {
      const table = document.querySelector('.results-table');
      if (!table) return;

      const headers = table.querySelectorAll<HTMLTableCellElement>('th[data-sort]');
      const tbody = table.querySelector('tbody');
      if (!tbody) return;

      headers.forEach(header => {
        const sortKey = header.dataset.sort;
        if (!sortKey) return;

        const button = header.querySelector('.sort-button');
        if (!button) return;

        button.addEventListener('click', () => {
          const currentDir = header.dataset.sortDir;
          const newDir = currentDir === 'asc' ? 'desc' : 'asc';

          headers.forEach(h => {
            if (h !== header) {
              delete h.dataset.sortDir;
            }
          });

          header.dataset.sortDir = newDir;

          const rows = Array.from(tbody.querySelectorAll('tr'));

          rows.sort((a, b) => {
            let aVal: string | number;
            let bVal: string | number;

            if (sortKey === 'metric') {
              const aCell = a.querySelector('.metric-name');
              const bCell = b.querySelector('.metric-name');
              aVal = aCell?.textContent ?? '';
              bVal = bCell?.textContent ?? '';
            } else {
              const aCell = a.querySelector(`td[data-column="${sortKey}"]`);
              const bCell = b.querySelector(`td[data-column="${sortKey}"]`);
              aVal = parseFloat(aCell?.getAttribute('data-value') ?? '0');
              bVal = parseFloat(bCell?.getAttribute('data-value') ?? '0');
            }

            if (typeof aVal === 'string' && typeof bVal === 'string') {
              return newDir === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
            }

            return newDir === 'asc'
              ? (aVal as number) - (bVal as number)
              : (bVal as number) - (aVal as number);
          });

          rows.forEach(row => tbody.appendChild(row));
        });
      });
    }

    it('should sort by metric name ascending', () => {
      initSorting();
      const metricButton = document.querySelector('th[data-sort="metric"] .sort-button') as HTMLButtonElement;
      metricButton?.click();

      const rows = document.querySelectorAll('tbody tr');
      const firstMetric = rows[0]?.querySelector('.metric-name')?.textContent;
      expect(firstMetric).toBe('Accuracy');
    });

    it('should sort by metric name descending on second click', () => {
      initSorting();
      const metricButton = document.querySelector('th[data-sort="metric"] .sort-button') as HTMLButtonElement;
      metricButton?.click();
      metricButton?.click();

      const rows = document.querySelectorAll('tbody tr');
      const firstMetric = rows[0]?.querySelector('.metric-name')?.textContent;
      expect(firstMetric).toBe('Precision');
    });

    it('should sort by numeric value', () => {
      initSorting();
      const bertBaseButton = document.querySelector('th[data-sort="bertBase"] .sort-button') as HTMLButtonElement;
      bertBaseButton?.click();

      const rows = document.querySelectorAll('tbody tr');
      const firstValue = rows[0]?.querySelector('td[data-column="bertBase"]')?.getAttribute('data-value');
      expect(firstValue).toBe('0.82'); // Lowest value first (ascending)
    });

    it('should toggle sort direction', () => {
      initSorting();
      const header = document.querySelector('th[data-sort="bertBase"]') as HTMLTableCellElement;
      const button = header?.querySelector('.sort-button') as HTMLButtonElement;

      button?.click();
      expect(header?.dataset.sortDir).toBe('asc');

      button?.click();
      expect(header?.dataset.sortDir).toBe('desc');
    });

    it('should clear other sort indicators when sorting a new column', () => {
      initSorting();
      const bertBaseHeader = document.querySelector('th[data-sort="bertBase"]') as HTMLTableCellElement;
      const bvBaseHeader = document.querySelector('th[data-sort="bvBase"]') as HTMLTableCellElement;
      const bertBaseButton = bertBaseHeader?.querySelector('.sort-button') as HTMLButtonElement;
      const bvBaseButton = bvBaseHeader?.querySelector('.sort-button') as HTMLButtonElement;

      bertBaseButton?.click();
      expect(bertBaseHeader?.dataset.sortDir).toBe('asc');

      bvBaseButton?.click();
      expect(bvBaseHeader?.dataset.sortDir).toBe('asc');
      expect(bertBaseHeader?.dataset.sortDir).toBeUndefined();
    });
  });

  describe('Data validation', () => {
    it('should have data-value attributes on metric cells', () => {
      const metricCells = document.querySelectorAll('.metric-value[data-value]');
      expect(metricCells.length).toBe(12); // 4 columns * 3 rows
    });

    it('should have valid numeric data-value attributes', () => {
      const metricCells = document.querySelectorAll('.metric-value[data-value]');
      metricCells.forEach(cell => {
        const value = parseFloat(cell.getAttribute('data-value') ?? '');
        expect(isNaN(value)).toBe(false);
        expect(value).toBeGreaterThanOrEqual(0);
        expect(value).toBeLessThanOrEqual(1);
      });
    });

    it('should have data-column attributes on metric cells', () => {
      const metricCells = document.querySelectorAll('.metric-value[data-column]');
      expect(metricCells.length).toBe(12);
    });
  });

  describe('Accessibility', () => {
    it('should have aria-label on sort buttons', () => {
      const sortButtons = document.querySelectorAll('.sort-button');
      sortButtons.forEach(button => {
        expect(button.getAttribute('aria-label')).toBeTruthy();
      });
    });

    it('should have scope attribute on header cells', () => {
      const headers = document.querySelectorAll('th');
      headers.forEach(header => {
        expect(header.getAttribute('scope')).toBe('col');
      });
    });
  });
});
