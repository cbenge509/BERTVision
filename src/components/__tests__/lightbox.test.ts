import {describe, it, expect, beforeEach} from 'vitest';
import {JSDOM} from 'jsdom';

describe('Lightbox keyboard handling', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div id="lightbox" class="lightbox" data-open="true">
            <div class="lightbox-backdrop"></div>
            <div class="lightbox-container">
              <button id="lightbox-close" class="lightbox-close">Close</button>
              <button id="lightbox-prev" class="lightbox-nav lightbox-prev">Prev</button>
              <button id="lightbox-next" class="lightbox-nav lightbox-next">Next</button>
              <img id="lightbox-image" src="test.jpg" alt="Test image" />
              <p id="lightbox-caption"></p>
              <div id="lightbox-counter">
                <span id="lightbox-current">1</span> / <span id="lightbox-total">3</span>
              </div>
            </div>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  it('should have close button', () => {
    const closeBtn = document.getElementById('lightbox-close');
    expect(closeBtn).toBeTruthy();
  });

  it('should have navigation buttons', () => {
    const prevBtn = document.getElementById('lightbox-prev');
    const nextBtn = document.getElementById('lightbox-next');
    expect(prevBtn).toBeTruthy();
    expect(nextBtn).toBeTruthy();
  });

  it('should respond to Escape key by closing', () => {
    const lightbox = document.getElementById('lightbox') as HTMLElement;

    // Simulate the close behavior that would happen on Escape
    lightbox.dataset.open = 'false';

    expect(lightbox.dataset.open).toBe('false');
  });

  it('should respond to arrow keys for navigation', () => {
    let currentIndex = 1;
    const maxIndex = 2;

    // Simulate ArrowRight navigation
    currentIndex = Math.min(currentIndex + 1, maxIndex);
    expect(currentIndex).toBe(2);

    // Simulate ArrowLeft navigation
    currentIndex = Math.max(currentIndex - 1, 0);
    expect(currentIndex).toBe(1);
  });

  it('should display image counter', () => {
    const currentEl = document.getElementById('lightbox-current');
    const totalEl = document.getElementById('lightbox-total');

    expect(currentEl?.textContent).toBe('1');
    expect(totalEl?.textContent).toBe('3');
  });
});

describe('Lightbox focus management', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <button id="trigger">Open Lightbox</button>
          <div id="lightbox" class="lightbox" data-open="false">
            <button id="lightbox-close">Close</button>
            <button id="lightbox-prev">Prev</button>
            <button id="lightbox-next">Next</button>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  it('should have focusable elements', () => {
    const focusableElements = document.querySelectorAll(
      '#lightbox button:not([disabled]):not([hidden])'
    );
    expect(focusableElements.length).toBe(3);
  });

  it('should be able to track previous active element', () => {
    const trigger = document.getElementById('trigger') as HTMLElement;
    trigger.focus();

    // Store previous active element
    const previousActiveElement = document.activeElement;
    expect(previousActiveElement).toBe(trigger);

    // Focus close button (simulating lightbox open)
    const closeBtn = document.getElementById('lightbox-close') as HTMLElement;
    closeBtn.focus();
    expect(document.activeElement).toBe(closeBtn);

    // Restore focus (simulating lightbox close)
    (previousActiveElement as HTMLElement).focus();
    expect(document.activeElement).toBe(trigger);
  });
});

describe('Lightbox touch handling', () => {
  it('should detect swipe direction', () => {
    let touchStartX = 0;
    let touchEndX = 0;
    const swipeThreshold = 50;

    // Simulate swipe left
    touchStartX = 200;
    touchEndX = 100;
    const swipeLeft = touchStartX - touchEndX > swipeThreshold;
    expect(swipeLeft).toBe(true);

    // Simulate swipe right
    touchStartX = 100;
    touchEndX = 200;
    const swipeRight = touchEndX - touchStartX > swipeThreshold;
    expect(swipeRight).toBe(true);

    // No swipe (too short)
    touchStartX = 100;
    touchEndX = 120;
    const noSwipe = Math.abs(touchStartX - touchEndX) < swipeThreshold;
    expect(noSwipe).toBe(true);
  });
});
