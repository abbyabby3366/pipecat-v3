/**
 * Dynamic Island Component
 * Provides an Apple-style interactive Dynamic Island for live web searching & search results.
 */

export class DynamicIsland {
  constructor(containerId) {
    this.container = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!this.container) {
      console.warn(`DynamicIsland: container element '${containerId}' not found, creating fallback.`);
      this.container = document.createElement('div');
      this.container.id = 'dynamic-island-container';
      document.body.prepend(this.container);
    }

    this.state = 'hidden'; // 'hidden' | 'searching' | 'results' | 'minimized'
    this.currentQuery = '';
    this.currentResults = [];
    this.currentType = 'web'; // 'web' | 'kb'
    this.autoCollapseTimer = null;
    this.remainingSeconds = 12;

    this.render();
    this.bindEvents();
  }

  render() {
    this.container.innerHTML = `
      <div id="dynamic-island" class="dynamic-island state-hidden" data-state="hidden">
        <!-- Compact / Searching View -->
        <div class="island-compact-view">
          <div class="island-lead-icon">
            <div class="radar-pulse"></div>
            <span class="icon-symbol" id="island-lead-icon-symbol">🌐</span>
          </div>

          <div class="island-searching-content">
            <span class="island-badge" id="island-searching-badge">正在搜索</span>
            <span class="island-query-preview" id="island-searching-query"></span>
          </div>

          <div class="island-searching-animation">
            <span class="wave-dot"></span>
            <span class="wave-dot"></span>
            <span class="wave-dot"></span>
          </div>

          <button id="island-expand-btn" class="island-mini-btn hidden" title="展开查看结果">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"></polyline></svg>
          </button>
        </div>

        <!-- Expanded Results Card View -->
        <div class="island-expanded-view">
          <div class="island-header">
            <div class="island-header-left">
              <span class="island-type-tag" id="island-expanded-tag">🌐 实时联网结果</span>
              <span class="island-query-tag" id="island-expanded-query"></span>
            </div>
            <div class="island-header-right">
              <span class="island-count-badge" id="island-count-badge">0 条结果</span>
              <button id="island-close-btn" class="island-action-btn" title="收起">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
              </button>
            </div>
          </div>

          <div class="island-results-scroll" id="island-results-list">
            <!-- Result cards injected here -->
          </div>

          <div class="island-footer">
            <div class="island-timer-bar-track">
              <div class="island-timer-bar-fill" id="island-timer-bar"></div>
            </div>
            <div class="island-footer-hint">
              <span>点击结果可在新标签页查看源网页</span>
            </div>
          </div>
        </div>
      </div>
    `;

    this.islandEl = document.getElementById('dynamic-island');
    this.leadIconSymbol = document.getElementById('island-lead-icon-symbol');
    this.searchingBadge = document.getElementById('island-searching-badge');
    this.searchingQuery = document.getElementById('island-searching-query');
    this.expandBtn = document.getElementById('island-expand-btn');

    this.expandedTag = document.getElementById('island-expanded-tag');
    this.expandedQuery = document.getElementById('island-expanded-query');
    this.countBadge = document.getElementById('island-count-badge');
    this.resultsList = document.getElementById('island-results-list');
    this.closeBtn = document.getElementById('island-close-btn');
    this.timerBar = document.getElementById('island-timer-bar');
  }

  bindEvents() {
    this.closeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.minimize();
    });

    this.expandBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      this.expand();
    });

    this.islandEl.addEventListener('click', (e) => {
      // Don't toggle if clicking a link
      if (e.target.closest('a') || e.target.closest('button')) return;
      if (this.state === 'minimized') {
        this.expand();
      } else if (this.state === 'results') {
        this.minimize();
      }
    });

    // Pause timer on hover
    this.islandEl.addEventListener('mouseenter', () => {
      if (this.timerBar) {
        this.timerBar.style.animationPlayState = 'paused';
      }
      if (this.autoCollapseTimer) {
        clearTimeout(this.autoCollapseTimer);
        this.autoCollapseTimer = null;
      }
    });

    this.islandEl.addEventListener('mouseleave', () => {
      if (this.state === 'results') {
        if (this.timerBar) {
          this.timerBar.style.animationPlayState = 'running';
        }
        this.startAutoCollapseTimer(6000);
      }
    });
  }

  /**
   * Display searching state in the Dynamic Island
   * @param {string} query Search query string
   * @param {'web'|'kb'} type Search type
   */
  showSearching(query, type = 'web') {
    this.currentQuery = query || '';
    this.currentType = type;
    this.state = 'searching';

    if (this.autoCollapseTimer) {
      clearTimeout(this.autoCollapseTimer);
      this.autoCollapseTimer = null;
    }

    const isKB = type === 'kb';
    this.leadIconSymbol.textContent = isKB ? '📚' : '🌐';
    this.searchingBadge.textContent = isKB ? '知识库检索中' : '实时联网搜索中';
    this.searchingQuery.textContent = query ? `"${query}"` : '正在检索最新数据...';
    this.expandBtn.classList.add('hidden');

    this.islandEl.className = 'dynamic-island state-searching';
    this.islandEl.dataset.state = 'searching';
  }

  /**
   * Display results in the expanded Dynamic Island
   * @param {string} query Search query string
   * @param {Array} results Array of search items {title, url, snippet, document, content, score}
   * @param {'web'|'kb'} type Search type
   */
  showResults(query, results = [], type = 'web') {
    this.currentQuery = query || this.currentQuery;
    this.currentResults = results || [];
    this.currentType = type;
    this.state = 'results';

    const isKB = type === 'kb';
    this.leadIconSymbol.textContent = isKB ? '📚' : '🌐';
    this.expandedTag.textContent = isKB ? '📚 ARK 专属知识库' : '🌐 Parallel 实时网页搜索';
    this.expandedQuery.textContent = this.currentQuery ? `"${this.currentQuery}"` : '';
    this.countBadge.textContent = `${this.currentResults.length} 条${isKB ? '文档' : '结果'}`;

    // Render results
    this.renderResultItems(this.currentResults, isKB);

    this.islandEl.className = 'dynamic-island state-results';
    this.islandEl.dataset.state = 'results';

    // Start auto collapse countdown to minimized pill
    this.startAutoCollapseTimer(12000);
  }

  renderResultItems(items, isKB) {
    if (!items || items.length === 0) {
      this.resultsList.innerHTML = `
        <div class="island-empty-state">
          <span>未获取到相关检索数据</span>
        </div>
      `;
      return;
    }

    this.resultsList.innerHTML = items
      .map((item, index) => {
        if (isKB) {
          const docName = item.document || '知识库文档';
          const score = item.score ? Math.round(item.score * 100) : null;
          const snippet = item.content || item.snippet || '';
          return `
            <div class="island-result-card kb-item" style="animation-delay: ${index * 60}ms">
              <div class="result-card-header">
                <div class="result-doc-title">
                  <span class="doc-icon">📄</span>
                  <strong>${this.escapeHtml(docName)}</strong>
                </div>
                ${score ? `<span class="result-score-badge">匹配度 ${score}%</span>` : ''}
              </div>
              <p class="result-snippet">${this.escapeHtml(snippet)}</p>
            </div>
          `;
        } else {
          const title = item.title || '网页搜索结果';
          const url = item.url || '#';
          const snippet = item.snippet || '';
          let domain = '';
          try {
            if (url && url !== '#') {
              domain = new URL(url).hostname.replace('www.', '');
            }
          } catch {
            domain = '';
          }

          return `
            <a href="${this.escapeHtml(url)}" target="_blank" rel="noopener noreferrer" class="island-result-card web-item" style="animation-delay: ${index * 60}ms">
              <div class="result-card-header">
                <div class="result-title-group">
                  ${domain ? `<span class="result-domain-chip">${this.escapeHtml(domain)}</span>` : ''}
                  <span class="result-title-text">${this.escapeHtml(title)}</span>
                </div>
                <div class="result-external-icon">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path><polyline points="15 3 21 3 21 9"></polyline><line x1="10" y1="14" x2="21" y2="3"></line></svg>
                </div>
              </div>
              ${snippet ? `<p class="result-snippet">${this.escapeHtml(snippet)}</p>` : ''}
            </a>
          `;
        }
      })
      .join('');
  }

  minimize() {
    if (this.state === 'hidden') return;
    this.state = 'minimized';

    if (this.autoCollapseTimer) {
      clearTimeout(this.autoCollapseTimer);
      this.autoCollapseTimer = null;
    }

    const isKB = this.currentType === 'kb';
    this.leadIconSymbol.textContent = isKB ? '📚' : '🌐';
    this.searchingBadge.textContent = isKB ? '知识库已匹配' : '搜索完成';
    this.searchingQuery.textContent = `${this.currentResults.length} 条相关结果 (点击查看)`;
    this.expandBtn.classList.remove('hidden');

    this.islandEl.className = 'dynamic-island state-minimized';
    this.islandEl.dataset.state = 'minimized';
  }

  expand() {
    if (this.currentResults && this.currentResults.length > 0) {
      this.showResults(this.currentQuery, this.currentResults, this.currentType);
    }
  }

  hide() {
    this.state = 'hidden';
    if (this.autoCollapseTimer) {
      clearTimeout(this.autoCollapseTimer);
      this.autoCollapseTimer = null;
    }
    this.islandEl.className = 'dynamic-island state-hidden';
    this.islandEl.dataset.state = 'hidden';
  }

  startAutoCollapseTimer(durationMs = 12000) {
    if (this.autoCollapseTimer) {
      clearTimeout(this.autoCollapseTimer);
    }
    if (this.timerBar) {
      this.timerBar.style.animation = 'none';
      void this.timerBar.offsetWidth; // Trigger reflow
      this.timerBar.style.animation = `islandTimerFill ${durationMs}ms linear forwards`;
    }

    this.autoCollapseTimer = setTimeout(() => {
      this.minimize();
    }, durationMs);
  }

  escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
}
