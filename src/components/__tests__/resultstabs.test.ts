/**
 * Unit tests for ResultsTabs component functionality.
 */

import {describe, it, expect, beforeEach, afterEach} from 'vitest';
import {JSDOM} from 'jsdom';

describe('ResultsTabs', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="results-tabs">
            <!-- Main category tabs -->
            <div class="tabs-container" role="tablist" aria-label="Result categories">
              <button
                type="button"
                role="tab"
                class="category-tab active"
                id="tab-glue"
                aria-selected="true"
                aria-controls="panel-glue"
                tabindex="0"
              >
                GLUE Benchmark
                <span class="tab-count">8</span>
              </button>
              <button
                type="button"
                role="tab"
                class="category-tab"
                id="tab-squad"
                aria-selected="false"
                aria-controls="panel-squad"
                tabindex="-1"
              >
                SQuAD 2.0
                <span class="tab-count">1</span>
              </button>
            </div>

            <!-- GLUE Panel -->
            <div
              role="tabpanel"
              class="category-panel active"
              id="panel-glue"
              aria-labelledby="tab-glue"
            >
              <div class="subtabs-wrapper">
                <div class="subtabs-container" role="tablist" aria-label="GLUE tasks">
                  <button type="button" role="tab" class="subtab active" id="subtab-cola" aria-selected="true" aria-controls="subpanel-cola" tabindex="0">CoLA</button>
                  <button type="button" role="tab" class="subtab" id="subtab-mnli" aria-selected="false" aria-controls="subpanel-mnli" tabindex="-1">MNLI</button>
                  <button type="button" role="tab" class="subtab" id="subtab-qnli" aria-selected="false" aria-controls="subpanel-qnli" tabindex="-1">QNLI</button>
                </div>
              </div>

              <div role="tabpanel" class="subpanel active" id="subpanel-cola" aria-labelledby="subtab-cola">CoLA content</div>
              <div role="tabpanel" class="subpanel" id="subpanel-mnli" aria-labelledby="subtab-mnli" hidden>MNLI content</div>
              <div role="tabpanel" class="subpanel" id="subpanel-qnli" aria-labelledby="subtab-qnli" hidden>QNLI content</div>
            </div>

            <!-- SQuAD Panel -->
            <div
              role="tabpanel"
              class="category-panel"
              id="panel-squad"
              aria-labelledby="tab-squad"
              hidden
            >
              SQuAD content
            </div>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  afterEach(() => {
    dom.window.close();
  });

  describe('Tab structure', () => {
    it('should have category tabs', () => {
      const categoryTabs = document.querySelectorAll('.category-tab');
      expect(categoryTabs.length).toBe(2);
    });

    it('should have sub-tabs for GLUE', () => {
      const subtabs = document.querySelectorAll('.subtab');
      expect(subtabs.length).toBe(3);
    });

    it('should have proper ARIA roles', () => {
      const tablist = document.querySelector('[role="tablist"]');
      expect(tablist).not.toBeNull();

      const tabs = document.querySelectorAll('[role="tab"]');
      expect(tabs.length).toBe(5); // 2 category + 3 subtabs

      const tabpanels = document.querySelectorAll('[role="tabpanel"]');
      expect(tabpanels.length).toBe(5); // 2 category + 3 subpanels
    });
  });

  describe('Tab state management', () => {
    function initTabs(): void {
      const tabs = document.querySelectorAll<HTMLButtonElement>('.category-tab');
      const panels = document.querySelectorAll('.category-panel');

      tabs.forEach(tab => {
        tab.addEventListener('click', () => {
          tabs.forEach(t => {
            t.classList.remove('active');
            t.setAttribute('aria-selected', 'false');
            t.setAttribute('tabindex', '-1');
          });

          tab.classList.add('active');
          tab.setAttribute('aria-selected', 'true');
          tab.setAttribute('tabindex', '0');

          const panelId = tab.getAttribute('aria-controls');
          panels.forEach(panel => {
            if (panel.id === panelId) {
              panel.classList.add('active');
              panel.removeAttribute('hidden');
            } else {
              panel.classList.remove('active');
              panel.setAttribute('hidden', '');
            }
          });
        });
      });

      // Also init subtabs
      const subtabs = document.querySelectorAll<HTMLButtonElement>('.subtab');
      const subpanels = document.querySelectorAll('.subpanel');

      subtabs.forEach(tab => {
        tab.addEventListener('click', () => {
          subtabs.forEach(t => {
            t.classList.remove('active');
            t.setAttribute('aria-selected', 'false');
            t.setAttribute('tabindex', '-1');
          });

          tab.classList.add('active');
          tab.setAttribute('aria-selected', 'true');
          tab.setAttribute('tabindex', '0');

          const panelId = tab.getAttribute('aria-controls');
          subpanels.forEach(panel => {
            if (panel.id === panelId) {
              panel.classList.add('active');
              panel.removeAttribute('hidden');
            } else {
              panel.classList.remove('active');
              panel.setAttribute('hidden', '');
            }
          });
        });
      });
    }

    it('should switch category tabs on click', () => {
      initTabs();
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;
      const gluePanel = document.querySelector('#panel-glue');
      const squadPanel = document.querySelector('#panel-squad');

      squadTab?.click();

      expect(squadTab.classList.contains('active')).toBe(true);
      expect(squadTab.getAttribute('aria-selected')).toBe('true');
      expect(gluePanel?.hasAttribute('hidden')).toBe(true);
      expect(squadPanel?.hasAttribute('hidden')).toBe(false);
    });

    it('should switch sub-tabs on click', () => {
      initTabs();
      const mnliTab = document.querySelector('#subtab-mnli') as HTMLButtonElement;
      const colaPanel = document.querySelector('#subpanel-cola');
      const mnliPanel = document.querySelector('#subpanel-mnli');

      mnliTab?.click();

      expect(mnliTab.classList.contains('active')).toBe(true);
      expect(mnliTab.getAttribute('aria-selected')).toBe('true');
      expect(colaPanel?.hasAttribute('hidden')).toBe(true);
      expect(mnliPanel?.hasAttribute('hidden')).toBe(false);
    });

    it('should update tabindex on tab switch', () => {
      initTabs();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      squadTab?.click();

      expect(glueTab.getAttribute('tabindex')).toBe('-1');
      expect(squadTab.getAttribute('tabindex')).toBe('0');
    });

    it('should deactivate previous tab when switching', () => {
      initTabs();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      squadTab?.click();

      expect(glueTab.classList.contains('active')).toBe(false);
      expect(glueTab.getAttribute('aria-selected')).toBe('false');
    });
  });

  describe('Keyboard navigation', () => {
    function setupKeyboardNav(): void {
      const tabs = document.querySelectorAll<HTMLButtonElement>('.category-tab');

      tabs.forEach(tab => {
        tab.addEventListener('keydown', (e: KeyboardEvent) => {
          const tabsArray = Array.from(tabs);
          const currentIndex = tabsArray.findIndex(t => t === document.activeElement);

          let newIndex = currentIndex;

          switch (e.key) {
            case 'ArrowRight':
            case 'ArrowDown':
              e.preventDefault();
              newIndex = currentIndex < tabsArray.length - 1 ? currentIndex + 1 : 0;
              break;
            case 'ArrowLeft':
            case 'ArrowUp':
              e.preventDefault();
              newIndex = currentIndex > 0 ? currentIndex - 1 : tabsArray.length - 1;
              break;
            case 'Home':
              e.preventDefault();
              newIndex = 0;
              break;
            case 'End':
              e.preventDefault();
              newIndex = tabsArray.length - 1;
              break;
          }

          const newTab = tabsArray[newIndex];
          if (newTab) {
            newTab.focus();
          }
        });
      });
    }

    it('should navigate to next tab with ArrowRight', () => {
      setupKeyboardNav();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      // Focus glue tab
      glueTab.focus();

      // Simulate ArrowRight
      const event = new dom.window.KeyboardEvent('keydown', {key: 'ArrowRight'});
      glueTab.dispatchEvent(event);

      expect(document.activeElement).toBe(squadTab);
    });

    it('should navigate to previous tab with ArrowLeft', () => {
      setupKeyboardNav();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      // Focus squad tab
      squadTab.focus();

      // Simulate ArrowLeft
      const event = new dom.window.KeyboardEvent('keydown', {key: 'ArrowLeft'});
      squadTab.dispatchEvent(event);

      expect(document.activeElement).toBe(glueTab);
    });

    it('should wrap around with ArrowRight at last tab', () => {
      setupKeyboardNav();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      // Focus squad tab (last)
      squadTab.focus();

      // Simulate ArrowRight
      const event = new dom.window.KeyboardEvent('keydown', {key: 'ArrowRight'});
      squadTab.dispatchEvent(event);

      expect(document.activeElement).toBe(glueTab);
    });

    it('should navigate to first tab with Home', () => {
      setupKeyboardNav();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      // Focus squad tab
      squadTab.focus();

      // Simulate Home
      const event = new dom.window.KeyboardEvent('keydown', {key: 'Home'});
      squadTab.dispatchEvent(event);

      expect(document.activeElement).toBe(glueTab);
    });

    it('should navigate to last tab with End', () => {
      setupKeyboardNav();
      const glueTab = document.querySelector('#tab-glue') as HTMLButtonElement;
      const squadTab = document.querySelector('#tab-squad') as HTMLButtonElement;

      // Focus glue tab
      glueTab.focus();

      // Simulate End
      const event = new dom.window.KeyboardEvent('keydown', {key: 'End'});
      glueTab.dispatchEvent(event);

      expect(document.activeElement).toBe(squadTab);
    });
  });

  describe('Accessibility', () => {
    it('should have aria-controls on tabs', () => {
      const tabs = document.querySelectorAll('[role="tab"]');
      tabs.forEach(tab => {
        expect(tab.getAttribute('aria-controls')).toBeTruthy();
      });
    });

    it('should have aria-labelledby on panels', () => {
      const panels = document.querySelectorAll('[role="tabpanel"]');
      panels.forEach(panel => {
        expect(panel.getAttribute('aria-labelledby')).toBeTruthy();
      });
    });

    it('should have aria-label on tablist', () => {
      const tablist = document.querySelector('[role="tablist"]');
      expect(tablist?.getAttribute('aria-label')).toBeTruthy();
    });

    it('should have matching aria-controls and panel ids', () => {
      const tabs = document.querySelectorAll('.category-tab');
      tabs.forEach(tab => {
        const controlsId = tab.getAttribute('aria-controls');
        const panel = document.getElementById(controlsId ?? '');
        expect(panel).not.toBeNull();
      });
    });
  });
});
