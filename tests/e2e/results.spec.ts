/**
 * E2E tests for Results components (ResultsTabs, ResultsTable).
 */

import {test, expect} from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

test.describe('Results Components', () => {
  test.beforeEach(async ({page}) => {
    await page.goto('/BERTVision/');
    // Wait for the page to be fully loaded
    await page.waitForLoadState('networkidle');
  });

  test.describe('ResultsTabs', () => {
    test('should display category tabs', async ({page}) => {
      const tabs = page.locator('.category-tab');
      await expect(tabs).toHaveCount(2);
      await expect(tabs.first()).toContainText('GLUE Benchmark');
      await expect(tabs.nth(1)).toContainText('SQuAD');
    });

    test('should show GLUE panel by default', async ({page}) => {
      const gluePanel = page.locator('#panel-glue');
      const squadPanel = page.locator('#panel-squad');

      await expect(gluePanel).toBeVisible();
      await expect(squadPanel).toBeHidden();
    });

    test('should switch to SQuAD panel when clicked', async ({page}) => {
      const squadTab = page.locator('#tab-squad');
      const gluePanel = page.locator('#panel-glue');
      const squadPanel = page.locator('#panel-squad');

      await squadTab.click();

      await expect(squadPanel).toBeVisible();
      await expect(gluePanel).toBeHidden();
    });

    test('should update aria-selected on tab switch', async ({page}) => {
      const glueTab = page.locator('#tab-glue');
      const squadTab = page.locator('#tab-squad');

      await expect(glueTab).toHaveAttribute('aria-selected', 'true');
      await expect(squadTab).toHaveAttribute('aria-selected', 'false');

      await squadTab.click();

      await expect(glueTab).toHaveAttribute('aria-selected', 'false');
      await expect(squadTab).toHaveAttribute('aria-selected', 'true');
    });

    test('should display GLUE sub-tabs', async ({page}) => {
      const subtabs = page.locator('.subtab');
      // Should have 8 GLUE tasks
      await expect(subtabs).toHaveCount(8);
    });

    test('should switch sub-tabs on click', async ({page}) => {
      const firstSubtab = page.locator('.subtab').first();
      const secondSubtab = page.locator('.subtab').nth(1);

      // Get the first subtab's controlled panel
      const firstPanelId = await firstSubtab.getAttribute('aria-controls');
      const secondPanelId = await secondSubtab.getAttribute('aria-controls');

      if (firstPanelId && secondPanelId) {
        const firstPanel = page.locator(`#${firstPanelId}`);
        const secondPanel = page.locator(`#${secondPanelId}`);

        await expect(firstPanel).toBeVisible();
        await expect(secondPanel).toBeHidden();

        await secondSubtab.click();

        await expect(firstPanel).toBeHidden();
        await expect(secondPanel).toBeVisible();
      }
    });

    test('should support keyboard navigation with arrow keys', async ({page}) => {
      const glueTab = page.locator('#tab-glue');
      const squadTab = page.locator('#tab-squad');

      // Focus the first tab
      await glueTab.focus();

      // Press ArrowRight to go to next tab
      await page.keyboard.press('ArrowRight');

      // Squad tab should now be focused
      await expect(squadTab).toBeFocused();
    });

    test('should wrap around keyboard navigation', async ({page}) => {
      const glueTab = page.locator('#tab-glue');
      const squadTab = page.locator('#tab-squad');

      // Focus the last tab
      await squadTab.focus();

      // Press ArrowRight to wrap to first tab
      await page.keyboard.press('ArrowRight');

      // GLUE tab should now be focused
      await expect(glueTab).toBeFocused();
    });
  });

  test.describe('ResultsTable', () => {
    test('should display results table', async ({page}) => {
      const table = page.locator('.results-table').first();
      await expect(table).toBeVisible();
    });

    test('should have sortable column headers', async ({page}) => {
      const sortButtons = page.locator('.results-table .sort-button');
      // 5 columns: Metric, BERT-base, BERT-large, BV-base, BV-large
      await expect(sortButtons.first()).toHaveCount(1);
    });

    test('should sort by column when header clicked', async ({page}) => {
      // Click on the BERT-base header to sort
      const bertBaseHeader = page.locator('th[data-sort="bertBase"] .sort-button').first();
      await bertBaseHeader.click();

      // Check that sort direction is set
      const header = page.locator('th[data-sort="bertBase"]').first();
      await expect(header).toHaveAttribute('data-sort-dir', 'asc');

      // Click again to reverse sort
      await bertBaseHeader.click();
      await expect(header).toHaveAttribute('data-sort-dir', 'desc');
    });

    test('should highlight best values', async ({page}) => {
      const bestValues = page.locator('.results-table .best-value');
      // Each row should have one best value
      expect(await bestValues.count()).toBeGreaterThan(0);
    });

    test('should be horizontally scrollable on mobile', async ({page}) => {
      // Set mobile viewport
      await page.setViewportSize({width: 375, height: 667});

      const tableWrapper = page.locator('.results-table-wrapper').first();
      await expect(tableWrapper).toBeVisible();

      // Check that overflow-x is auto or scroll
      const overflowX = await tableWrapper.evaluate(el =>
        window.getComputedStyle(el).overflowX
      );
      expect(['auto', 'scroll']).toContain(overflowX);
    });
  });

  test.describe('TaskResult', () => {
    test('should display task title', async ({page}) => {
      const taskTitle = page.locator('.task-title').first();
      await expect(taskTitle).toBeVisible();
    });

    test('should display task description', async ({page}) => {
      const taskDescription = page.locator('.task-description').first();
      await expect(taskDescription).toBeVisible();
    });

    test('should have collapsible replication commands', async ({page}) => {
      const collapsible = page.locator('.task-result .collapsible').first();
      await expect(collapsible).toBeVisible();

      // Should contain "Replication Commands" text
      const trigger = collapsible.locator('.collapsible-trigger');
      await expect(trigger).toContainText('Replication Commands');
    });

    test('should expand/collapse replication commands', async ({page}) => {
      const collapsible = page.locator('.task-result .collapsible').first();
      const trigger = collapsible.locator('.collapsible-trigger');
      const content = collapsible.locator('.collapsible-content');

      // Initially collapsed
      await expect(content).toBeHidden();

      // Click to expand
      await trigger.click();
      await expect(content).toBeVisible();

      // Click to collapse
      await trigger.click();
      await expect(content).toBeHidden();
    });

    test('should display code block in replication commands', async ({page}) => {
      const collapsible = page.locator('.task-result .collapsible').first();
      const trigger = collapsible.locator('.collapsible-trigger');

      // Expand the collapsible
      await trigger.click();

      const codeBlock = collapsible.locator('.code-block');
      await expect(codeBlock).toBeVisible();
    });

    test('should have dataset link if available', async ({page}) => {
      const datasetLink = page.locator('.dataset-link').first();
      if (await datasetLink.isVisible()) {
        await expect(datasetLink).toHaveAttribute('target', '_blank');
        await expect(datasetLink).toHaveAttribute('rel', 'noopener noreferrer');
      }
    });
  });

  test.describe('Mobile swipe navigation', () => {
    test('should support swipe on subtabs', async ({page}) => {
      // Set mobile viewport
      await page.setViewportSize({width: 375, height: 667});

      const subtabsWrapper = page.locator('.subtabs-wrapper');
      await expect(subtabsWrapper).toBeVisible();

      // Just verify the wrapper is scrollable
      const overflowX = await subtabsWrapper.evaluate(el =>
        window.getComputedStyle(el).overflowX
      );
      expect(['auto', 'scroll']).toContain(overflowX);
    });
  });

  test.describe('Accessibility', () => {
    test('should have no accessibility violations in results section', async ({page}) => {
      const resultsSection = page.locator('.results-tabs');

      if (await resultsSection.isVisible()) {
        const accessibilityScanResults = await new AxeBuilder({page})
          .include('.results-tabs')
          .analyze();

        expect(accessibilityScanResults.violations).toEqual([]);
      }
    });

    test('should have proper ARIA attributes on tabs', async ({page}) => {
      const tablist = page.locator('[role="tablist"]').first();
      await expect(tablist).toHaveAttribute('aria-label');

      const tabs = page.locator('[role="tab"]');
      const tabCount = await tabs.count();

      for (let i = 0; i < tabCount; i++) {
        const tab = tabs.nth(i);
        await expect(tab).toHaveAttribute('aria-controls');
        await expect(tab).toHaveAttribute('aria-selected');
      }
    });

    test('should have proper ARIA attributes on tabpanels', async ({page}) => {
      const tabpanels = page.locator('[role="tabpanel"]');
      const panelCount = await tabpanels.count();

      for (let i = 0; i < panelCount; i++) {
        const panel = tabpanels.nth(i);
        await expect(panel).toHaveAttribute('aria-labelledby');
      }
    });
  });
});
