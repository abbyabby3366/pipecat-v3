/**
 * Main Application Controller
 * Manages PipecatClient WebRTC session, scrollable conversation history, and audio visualizer.
 */

import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';
import {
  createIcons,
  Activity,
  Info,
  Phone,
  PhoneOff,
  Mic,
  MicOff,
  Volume2,
  X,
  Terminal,
  Zap,
  Database,
  Cpu,
  Sparkles,
} from 'lucide';

import { AuroraBackground } from './components/aurora-background.js';
import { ReactiveOrb } from './components/reactive-orb.js';
import { AudioVisualizer } from './components/audio-visualizer.js';
import { DynamicIsland } from './components/dynamic-island.js';

// DOM Elements
const appLayout = document.getElementById('app-layout');
const callBtn = document.getElementById('call-btn');
const endCallBtn = document.getElementById('end-call-btn');
const micBtn = document.getElementById('mic-btn');
const micLabel = document.getElementById('mic-label');
const startCallContainer = document.getElementById('start-call-container');
const connectedControlsContainer = document.getElementById('connected-controls-container');

const conversationStream = document.getElementById('conversation-stream');
const conversationMessages = document.getElementById('conversation-messages');

const connectionBadge = document.getElementById('connection-badge');
const connectionStatusText = document.getElementById('connection-status-text');
const agentStatePill = document.getElementById('agent-state-pill');
const agentStateText = document.getElementById('agent-state-text');
const orbSpokenText = document.getElementById('orb-spoken-text');
const botAudio = document.getElementById('bot-audio');

// Drawer & Tab Elements
const sideDrawer = document.getElementById('side-drawer');
const toggleDrawerBtn = document.getElementById('toggle-drawer-btn');
const closeDrawerBtn = document.getElementById('close-drawer-btn');
const toggleMetricsBtn = document.getElementById('toggle-metrics-btn');
const drawerTabBtns = document.querySelectorAll('.drawer-tab-btn');
const drawerTabPanes = document.querySelectorAll('.drawer-tab-pane');
const toolActivityFeed = document.getElementById('tool-activity-feed');

const metricStt = document.getElementById('metric-stt');
const metricLlm = document.getElementById('metric-llm');
const metricTts = document.getElementById('metric-tts');
const metricTools = document.getElementById('metric-tools');
const e2eBadge = document.getElementById('e2e-badge');

// Voice Speed Controls
const voiceSpeedSlider = document.getElementById('voice-speed-slider');
const voiceSpeedBadge = document.getElementById('voice-speed-badge');
const speedChips = document.querySelectorAll('.speed-chip');

// Global State
let client = null;
let isConnected = false;
let isMuted = false;
let hasSphereMovedDown = false;
let currentBotMsgElement = null;
let currentBotText = '';
let wakeLock = null;
let activeDrawerTab = 'metrics';
let toolStartTime = null;
let currentVoiceSpeed = parseFloat(localStorage.getItem('voice_speed') || '1.0');

async function requestWakeLock() {
  try {
    if ('wakeLock' in navigator) {
      wakeLock = await navigator.wakeLock.request('screen');
      wakeLock.addEventListener('release', () => {
        wakeLock = null;
      });
    }
  } catch (err) {
    console.debug('Wake Lock request skipped:', err);
  }
}

function releaseWakeLock() {
  if (wakeLock) {
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }
}

// Initialize Visual Elements
const aurora = new AuroraBackground('aurora-canvas');
const orb = new ReactiveOrb('orb-canvas', { hue: 6.0 });
const visualizer = new AudioVisualizer(orb);
const dynamicIsland = new DynamicIsland('dynamic-island-container');

// Initialize Lucide Icons
function refreshIcons() {
  createIcons({
    icons: {
      Activity,
      Info,
      Phone,
      PhoneOff,
      Mic,
      MicOff,
      Volume2,
      X,
      Terminal,
      Zap,
      Database,
      Cpu,
      Sparkles,
    },
  });
}
refreshIcons();

// State Helper
function setAgentState(state, text) {
  orb.setState(state);
  agentStatePill.dataset.state = state;
  agentStateText.textContent = text;

  const iconMap = {
    idle: '✨',
    listening: '🎙️',
    thinking: '🧠',
    speaking: '🔊',
  };
  const iconSpan = agentStatePill.querySelector('.state-icon');
  if (iconSpan) iconSpan.textContent = iconMap[state] || '✨';
}

function setConnectionState(state, label) {
  connectionBadge.dataset.state = state;
  connectionStatusText.textContent = label;
}

function setOrbText(text) {
  if (!text || !text.trim()) return;
  orbSpokenText.textContent = text.trim();
}

function moveSphereDown() {
  if (!hasSphereMovedDown) {
    hasSphereMovedDown = true;
    appLayout.classList.add('sphere-down');
    conversationStream.classList.remove('hidden');
  }
}

// Append Chat Message Bubble to the Main Scrollable Stream
function appendChatMessage(role, text) {
  moveSphereDown();
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.textContent = text;

  conversationMessages.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function scrollToBottom() {
  conversationStream.scrollTop = conversationStream.scrollHeight;
}

// Update Metric UI Helpers
function updateMetricValue(element, val) {
  if (!element || val === undefined || val === null || isNaN(val)) return;
  element.textContent = `${Math.round(val)} ms`;
}

function updateE2EBadge(val) {
  if (!e2eBadge || val === undefined || val === null || isNaN(val)) return;
  const rounded = Math.round(val);
  e2eBadge.textContent = `总延迟: ${rounded} ms`;
  if (rounded < 900) {
    e2eBadge.className = 'e2e-badge fast';
  } else {
    e2eBadge.className = 'e2e-badge';
  }
}

// Drawer & Tab Switcher Management
function switchDrawerTab(tabId) {
  activeDrawerTab = tabId;
  drawerTabBtns.forEach((btn) => {
    if (btn.dataset.tab === tabId) {
      btn.classList.add('active');
    } else {
      btn.classList.remove('active');
    }
  });

  drawerTabPanes.forEach((pane) => {
    if (pane.id === `tab-pane-${tabId}`) {
      pane.classList.remove('hidden');
      pane.classList.add('active');
    } else {
      pane.classList.add('hidden');
      pane.classList.remove('active');
    }
  });

  // Update Header Button active highlights
  if (tabId === 'metrics') {
    toggleMetricsBtn.classList.add('active');
    toggleDrawerBtn.classList.remove('active');
  } else {
    toggleDrawerBtn.classList.add('active');
    toggleMetricsBtn.classList.remove('active');
  }
}

function toggleDrawer(targetTab = null, forceOpen = null) {
  const isCurrentlyOpen = !sideDrawer.classList.contains('closed');
  let shouldOpen = forceOpen !== null ? forceOpen : !isCurrentlyOpen;

  // If clicking the same button when already open with that tab, toggle close
  if (targetTab && isCurrentlyOpen && activeDrawerTab === targetTab && forceOpen === null) {
    shouldOpen = false;
  }

  if (shouldOpen) {
    if (targetTab) {
      switchDrawerTab(targetTab);
    }
    sideDrawer.classList.remove('closed');
  } else {
    sideDrawer.classList.add('closed');
    toggleMetricsBtn.classList.remove('active');
    toggleDrawerBtn.classList.remove('active');
  }
}

// Tab navigation button click listeners
drawerTabBtns.forEach((btn) => {
  btn.addEventListener('click', () => {
    const tab = btn.dataset.tab;
    if (tab) switchDrawerTab(tab);
  });
});

toggleMetricsBtn.addEventListener('click', () => toggleDrawer('metrics'));
toggleDrawerBtn.addEventListener('click', () => toggleDrawer('tools'));
closeDrawerBtn.addEventListener('click', () => toggleDrawer(null, false));

// Voice Speed Management
function getSpeedLabel(speed) {
  if (speed <= 0.85) return `${speed.toFixed(2)}x 沉稳`;
  if (speed >= 0.95 && speed <= 1.05) return '1.0x 标准';
  if (speed > 1.05 && speed <= 1.25) return `${speed.toFixed(2)}x 推荐`;
  return `${speed.toFixed(2)}x 极速`;
}

function setVoiceSpeed(speed, fromSlider = false) {
  const clamped = Math.max(0.6, Math.min(1.5, parseFloat(speed)));
  currentVoiceSpeed = clamped;
  localStorage.setItem('voice_speed', clamped.toString());

  if (voiceSpeedBadge) {
    voiceSpeedBadge.textContent = getSpeedLabel(clamped);
  }

  if (voiceSpeedSlider && !fromSlider) {
    voiceSpeedSlider.value = clamped.toString();
  }

  speedChips.forEach((chip) => {
    const chipSpeed = parseFloat(chip.dataset.speed || '1.0');
    if (Math.abs(chipSpeed - clamped) < 0.04) {
      chip.classList.add('active');
    } else {
      chip.classList.remove('active');
    }
  });

  // If connected, send dynamic update to bot
  if (client && isConnected) {
    try {
      client.sendClientMessage('set-voice-speed', { speed: clamped });
    } catch (err) {
      console.warn('Could not send voice speed update:', err);
    }
  }
}

// Initialize Voice Speed Controls
if (voiceSpeedSlider) {
  voiceSpeedSlider.value = currentVoiceSpeed.toString();
  voiceSpeedSlider.addEventListener('input', (e) => {
    setVoiceSpeed(e.target.value, true);
  });
}

speedChips.forEach((chip) => {
  chip.addEventListener('click', () => {
    const speed = parseFloat(chip.dataset.speed || '1.0');
    setVoiceSpeed(speed);
  });
});
setVoiceSpeed(currentVoiceSpeed);

// Connect / Start WebRTC Call
async function startCall() {
  callBtn.disabled = true;
  setConnectionState('connecting', '连接中...');
  setAgentState('thinking', '正在建立语音连接...');
  setOrbText('连接中...');

  try {
    const resp = await fetch(`${window.location.origin}/api/connect`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        voice_speed: currentVoiceSpeed,
      }),
    });
    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error: ${resp.status}`);
    }
    const { room_url, token } = await resp.json();

    client = new PipecatClient({
      transport: new DailyTransport(),
      enableMic: true,
      enableCam: false,
    });

    // 1. Client Lifecycle
    client.on(RTVIEvent.BotConnected, () => {
      setConnectionState('connected', '已连接');
      setAgentState('idle', '请说话...');
      setOrbText('嘿，你好');

      // Switch to connected controls view
      startCallContainer.classList.add('hidden');
      connectedControlsContainer.classList.remove('hidden');
      isConnected = true;
      requestWakeLock();
    });

    client.on(RTVIEvent.Disconnected, () => {
      handleDisconnect();
    });

    // 2. Audio Track Management & Playback
    client.on(RTVIEvent.TrackStarted, (track, participant) => {
      if (track.kind === 'audio' && !participant?.local) {
        botAudio.srcObject = new MediaStream([track]);
        botAudio.play().catch((err) => console.warn('botAudio play error:', err));
        visualizer.attachStream(botAudio.srcObject, 'speaker');
      }
    });

    client.on(RTVIEvent.Error, (err) => {
      console.error('RTVI Error:', err);
    });

    client.on(RTVIEvent.DeviceError, (err) => {
      console.error('RTVI Device Error:', err);
    });

    // 3. Speaking Turn Events
    let currentUserMsgElement = null;
    let currentUserText = '';

    client.on(RTVIEvent.UserStartedSpeaking, () => {
      setAgentState('listening', '正在聆听...');
      currentBotMsgElement = null;
      currentBotText = '';
      currentUserMsgElement = null;
      currentUserText = '';
      moveSphereDown();
    });

    client.on(RTVIEvent.UserStoppedSpeaking, () => {
      setAgentState('thinking', '正在思考...');
      currentUserMsgElement = null;
      currentUserText = '';
    });

    client.on(RTVIEvent.BotLlmStarted, () => {
      currentBotMsgElement = null;
      currentBotText = '';
    });

    client.on(RTVIEvent.BotStoppedSpeaking, () => {
      setAgentState('idle', '请说话...');
      currentBotMsgElement = null;
      currentBotText = '';
    });

    // 4. Real-time Transcripts (Chat Bubbles in Scrollable Stream)
    client.on(RTVIEvent.UserTranscript, (data) => {
      if (data?.text) {
        moveSphereDown();
        const text = data.text.trim();
        if (!text) return;

        if (data.final) {
          if (!currentUserMsgElement) {
            currentUserText = text;
            currentUserMsgElement = appendChatMessage('user', currentUserText);
          } else {
            currentUserText = currentUserText ? `${currentUserText} ${text}` : text;
            currentUserMsgElement.textContent = currentUserText;
            scrollToBottom();
          }
        }
      }
    });

    // Stream LLM tokens live character-by-character into chat bubble
    client.on(RTVIEvent.BotLlmText, (data) => {
      if (data?.text) {
        moveSphereDown();
        currentBotText += data.text;

        if (!currentBotMsgElement) {
          currentBotMsgElement = appendChatMessage('bot', currentBotText);
        } else {
          currentBotMsgElement.textContent = currentBotText;
          scrollToBottom();
        }
      }
    });

    // Fallback for non-LLM bot transcripts (e.g. system speech)
    client.on(RTVIEvent.BotTranscript, (data) => {
      if (data?.text && !currentBotText) {
        moveSphereDown();
        currentBotText = data.text;
        currentBotMsgElement = appendChatMessage('bot', currentBotText);
      }
    });

    // 5. Function Calling / Tools Notification & Dynamic Island
    let activeToolBubble = null;
    let currentSearchQuery = '';
    let currentSearchType = 'web';

    const handleToolCallStarted = (call) => {
      toolStartTime = performance.now();
      const fnName = call?.function_name || 'Tool';
      const isKB = fnName.includes('knowledge');
      const query = call?.args?.query || call?.arguments?.query || '';
      currentSearchQuery = query;
      currentSearchType = isKB ? 'kb' : 'web';
      const actionLabel = isKB ? '正在检索专属知识库' : '正在联网搜索';
      const typeClass = isKB ? 'knowledge-base' : 'web-search';
      const icon = isKB ? '📚' : '🌐';

      setAgentState('thinking', isKB ? '检索专属知识库...' : '实时联网搜索...');
      setOrbText(isKB ? '正在检索知识库...' : '正在网上搜索...');

      // Trigger Apple-style Dynamic Island at the top of the screen!
      dynamicIsland.showSearching(query, currentSearchType);

      // Append visual status card directly into active conversation stream
      moveSphereDown();
      const toolBubble = document.createElement('div');
      toolBubble.className = `chat-bubble tool-status ${typeClass}`;
      toolBubble.innerHTML = `
        <div class="tool-status-spinner"></div>
        <div class="tool-status-text">
          <span>${icon} ${actionLabel}${query ? `: <span class="tool-status-query">"${query}"</span>` : '...'}</span>
        </div>
      `;
      conversationMessages.appendChild(toolBubble);
      scrollToBottom();
      activeToolBubble = toolBubble;
    };

    client.on(RTVIEvent.LLMFunctionCall, handleToolCallStarted);
    client.on(RTVIEvent.LLMFunctionCallStarted, handleToolCallStarted);
    client.on(RTVIEvent.LLMFunctionCallInProgress, handleToolCallStarted);

    // Display search results on the Dynamic Island and Telemetry Feed when search completes
    client.on(RTVIEvent.LLMFunctionCallStopped, (data) => {
      const durationMs = toolStartTime ? Math.round(performance.now() - toolStartTime) : null;
      if (durationMs !== null) {
        updateMetricValue(metricTools, durationMs);
      }

      const fnName = data?.function_name || '';
      const isKB = fnName.includes('knowledge') || currentSearchType === 'kb';
      const result = data?.result;
      let resultCount = 0;
      let snippetPreview = '';

      if (result) {
        if (result.results && Array.isArray(result.results)) {
          dynamicIsland.showResults(currentSearchQuery, result.results, 'web');
          resultCount = result.results.length;
          if (resultCount > 0 && result.results[0].title) {
            snippetPreview = result.results[0].title;
          }
        } else if (result.records && Array.isArray(result.records)) {
          dynamicIsland.showResults(currentSearchQuery, result.records, 'kb');
          resultCount = result.records.length;
          if (resultCount > 0 && result.records[0].document) {
            snippetPreview = `${result.records[0].document} (匹配度: ${result.records[0].score || ''})`;
          }
        }
      }

      // Append card to telemetry feed
      const card = document.createElement('div');
      card.className = `tool-event-card ${isKB ? 'kb' : 'search'}`;
      card.innerHTML = `
        <div class="tool-event-header">
          <div class="tool-badge ${isKB ? 'kb' : ''}">
            <span>${isKB ? '📚 Misuedi ARK 知识库' : '🌐 Parallel 实时搜索'}</span>
          </div>
          ${durationMs ? `<span class="tool-time-badge">${durationMs} ms</span>` : ''}
        </div>
        <div class="tool-query-text">
          查询关键词: <em>${currentSearchQuery || fnName}</em>
        </div>
        ${
          snippetPreview
            ? `<div class="tool-result-preview">
                 <span>匹配 ${resultCount} 条结果: ${snippetPreview}</span>
               </div>`
            : ''
        }
      `;
      toolActivityFeed.appendChild(card);
    });

    client.on(RTVIEvent.BotStartedSpeaking, () => {
      setAgentState('speaking', '正在回答...');
      moveSphereDown();
      if (activeToolBubble) {
        const spinner = activeToolBubble.querySelector('.tool-status-spinner');
        if (spinner) {
          spinner.style.display = 'none';
        }
        activeToolBubble = null;
      }
    });

    // 6. Metrics & Latency Telemetry (RTVI Protocol Metrics)
    client.on(RTVIEvent.Metrics, (metrics) => {
      if (!metrics) return;

      // Handle TTFB metrics array
      if (Array.isArray(metrics.ttfb)) {
        metrics.ttfb.forEach((item) => {
          const proc = item.processor || '';
          const ms = Math.round((item.value || 0) * 1000);
          if (proc.includes('TTS') || proc.includes('CartesiaTTS')) {
            updateMetricValue(metricTts, ms);
          } else if (proc.includes('LLM') || proc.includes('OpenRouter') || proc.includes('Cerebras')) {
            updateMetricValue(metricLlm, ms);
          }
        });
      } else if (typeof metrics.ttfb === 'number') {
        updateMetricValue(metricTts, Math.round(metrics.ttfb * 1000));
      }

      // Handle Processing metrics array (e.g. STT)
      if (Array.isArray(metrics.processing)) {
        metrics.processing.forEach((item) => {
          const proc = item.processor || '';
          const ms = Math.round((item.value || 0) * 1000);
          if (proc.includes('STT') || proc.includes('CartesiaSTT')) {
            updateMetricValue(metricStt, ms);
          }
        });
      }
    });

    // 7. Custom Server-side Latency Breakdown & Telemetry Messages
    const handleServerMessage = (data) => {
      if (!data) return;
      if (data.type === 'latency_report') {
        if (data.stt_ms !== undefined && data.stt_ms !== null) {
          updateMetricValue(metricStt, data.stt_ms);
        }
        if (data.llm_ms !== undefined && data.llm_ms !== null) {
          updateMetricValue(metricLlm, data.llm_ms);
        }
        if (data.tts_ms !== undefined && data.tts_ms !== null) {
          updateMetricValue(metricTts, data.tts_ms);
        }
        if (data.tools_ms !== undefined && data.tools_ms !== null) {
          updateMetricValue(metricTools, data.tools_ms);
        }
        if (data.total_latency_ms !== undefined && data.total_latency_ms !== null) {
          updateE2EBadge(data.total_latency_ms);
        }
      }
    };

    client.on(RTVIEvent.ServerMessage, handleServerMessage);
    client.on('server-message', handleServerMessage);

    // Connect to Daily room using credentials
    await client.connect({
      url: room_url,
      token: token,
    });

  } catch (err) {
    console.error('Connection failed:', err);
    setOrbText('连接失败');
    handleDisconnect();
  }
}

// End Call
async function endCall() {
  setConnectionState('connecting', '正在挂断...');
  try {
    await client?.disconnect();
  } finally {
    handleDisconnect();
  }
}

function handleDisconnect() {
  releaseWakeLock();
  isConnected = false;
  hasSphereMovedDown = false;
  dynamicIsland.hide();
  appLayout.classList.remove('sphere-down');
  conversationStream.classList.add('hidden');
  conversationMessages.innerHTML = '';

  startCallContainer.classList.remove('hidden');
  connectedControlsContainer.classList.add('hidden');
  callBtn.disabled = false;
  isMuted = false;
  micLabel.textContent = '静音';
  micBtn.classList.remove('active-mute');
  micBtn.innerHTML = '<i data-lucide="mic"></i>';

  setConnectionState('disconnected', '就绪');
  setAgentState('idle', '准备就绪');
  setOrbText('嘿，你好');

  visualizer.detach();
  if (botAudio.srcObject) {
    botAudio.srcObject = null;
  }
  client = null;
  refreshIcons();
}

// Mic Toggle
micBtn.addEventListener('click', () => {
  if (!client || !isConnected) return;
  isMuted = !isMuted;
  client.enableMic(!isMuted);

  if (isMuted) {
    micBtn.classList.add('active-mute');
    micBtn.innerHTML = '<i data-lucide="mic-off"></i>';
    micLabel.textContent = '已静音';
  } else {
    micBtn.classList.remove('active-mute');
    micBtn.innerHTML = '<i data-lucide="mic"></i>';
    micLabel.textContent = '静音';
  }
  refreshIcons();
});

// Call Buttons
callBtn.addEventListener('click', startCall);
endCallBtn.addEventListener('click', endCall);
