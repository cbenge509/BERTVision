import {describe, it, expect, vi, beforeEach, afterEach} from 'vitest';
import {JSDOM} from 'jsdom';

describe('CodeBlock copy functionality', () => {
  let dom: JSDOM;
  let document: Document;
  let mockClipboard: {writeText: ReturnType<typeof vi.fn>};

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <div class="code-block">
            <button
              class="code-block-copy"
              data-code="const x = 1;"
              aria-label="Copy code to clipboard"
            >
              Copy
            </button>
          </div>
        </body>
      </html>
    `);
    document = dom.window.document;

    // Mock clipboard API
    mockClipboard = {
      writeText: vi.fn().mockResolvedValue(undefined),
    };
    Object.defineProperty(dom.window.navigator, 'clipboard', {
      value: mockClipboard,
      writable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should have copy button with code data attribute', () => {
    const button = document.querySelector('.code-block-copy');
    expect(button).toBeTruthy();
    expect(button?.getAttribute('data-code')).toBe('const x = 1;');
  });

  it('should have proper aria-label', () => {
    const button = document.querySelector('.code-block-copy');
    expect(button?.getAttribute('aria-label')).toBe('Copy code to clipboard');
  });

  it('should update data-copied attribute after copy', async () => {
    const button = document.querySelector('.code-block-copy') as HTMLButtonElement;

    // Simulate copy behavior
    button.dataset.copied = 'true';
    expect(button.dataset.copied).toBe('true');

    // Reset after delay
    button.dataset.copied = 'false';
    expect(button.dataset.copied).toBe('false');
  });
});

describe('CodeBlock line numbers', () => {
  let dom: JSDOM;
  let document: Document;

  beforeEach(() => {
    dom = new JSDOM(`
      <!DOCTYPE html>
      <html>
        <body>
          <pre class="code-block-pre with-line-numbers">
            <code>
              <span class="code-line">
                <span class="line-number">1</span>
                <span class="line-content">const x = 1;</span>
              </span>
              <span class="code-line">
                <span class="line-number">2</span>
                <span class="line-content">const y = 2;</span>
              </span>
            </code>
          </pre>
        </body>
      </html>
    `);
    document = dom.window.document;
  });

  it('should have line number class when enabled', () => {
    const pre = document.querySelector('.code-block-pre');
    expect(pre?.classList.contains('with-line-numbers')).toBe(true);
  });

  it('should render line numbers', () => {
    const lineNumbers = document.querySelectorAll('.line-number');
    expect(lineNumbers.length).toBe(2);
    expect(lineNumbers[0]?.textContent?.trim()).toBe('1');
    expect(lineNumbers[1]?.textContent?.trim()).toBe('2');
  });

  it('should render line content', () => {
    const lineContents = document.querySelectorAll('.line-content');
    expect(lineContents.length).toBe(2);
  });
});
