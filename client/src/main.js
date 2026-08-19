/**
 * Main Application Controller
 * Manages PipecatClient WebRTC session, scrollable conversation history, and audio visualizer.
 */

import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { SmallWebRTCTransport } from '@pipecat-ai/small-webrtc-transport';
import { createIcons, Activity, Info, Phone, PhoneOff, Mic, MicOff, Volume2, X, Terminal } from 'lucide';

import { AuroraBackground } from './components/aurora-background.js';
import { ReactiveOrb } from './components/reactive-orb.js';
import { AudioVisualizer } from './components/audio-visualizer.js';

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

// Drawer & Metrics Elements
const sideDrawer = document.getElementById('side-drawer');
const toggleDrawerBtn = document.getElementById('toggle-drawer-btn');
const closeDrawerBtn = document.getElementById('close-drawer-btn');
const toggleMetricsBtn = document.getElementById('toggle-metrics-btn');
const toolActivityFeed = document.getElementById('tool-activity-feed');

const metricStt = document.getElementById('metric-stt');
const metricLlm = document.getElementById('metric-llm');
const metricTts = document.getElementById('metric-tts');
const metricTools = document.getElementById('metric-tools');
const e2eBadge = document.getElementById('e2e-badge');

// Global State
let client = null;
let isConnected = false;
let isMuted = false;
let hasSphereMovedDown = false;
let currentBotMsgElement = null;
let currentBotText = '';

// Initialize Visual Elements
const aurora = new AuroraBackground('aurora-canvas');
const orb = new ReactiveOrb('orb-canvas', { hue: 6.0 });
const visualizer = new AudioVisualizer(orb);

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

// Drawer Toggle
function toggleDrawer(forceOpen = null) {
  const shouldOpen = forceOpen !== null ? forceOpen : sideDrawer.classList.contains('closed');
  if (shouldOpen) {
    sideDrawer.classList.remove('closed');
  } else {
    sideDrawer.classList.add('closed');
  }
}

toggleDrawerBtn.addEventListener('click', () => toggleDrawer());
closeDrawerBtn.addEventListener('click', () => toggleDrawer(false));
toggleMetricsBtn.addEventListener('click', () => toggleDrawer(true));

// Connect / Start WebRTC Call
async function startCall() {
  callBtn.disabled = true;
  setConnectionState('connecting', '连接中...');
  setAgentState('thinking', '正在建立语音连接...');
  setOrbText('连接中...');

  try {
    client = new PipecatClient({
      transport: new SmallWebRTCTransport(),
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
    });

    client.on(RTVIEvent.Disconnected, () => {
      handleDisconnect();
    });

    // 2. Audio Track Management & Playback
    client.on(RTVIEvent.TrackStarted, (track, participant) => {
      if (track.kind === 'audio' && !participant?.local) {
        botAudio.srcObject = new MediaStream([track]);
        visualizer.attachStream(botAudio.srcObject, 'speaker');
      }
    });

    // 3. Speaking Turn Events
    client.on(RTVIEvent.UserStartedSpeaking, () => {
      setAgentState('listening', '正在聆听...');
      currentBotMsgElement = null;
      currentBotText = '';
      moveSphereDown();
    });

    client.on(RTVIEvent.UserStoppedSpeaking, () => {
      setAgentState('thinking', '正在思考...');
    });

    client.on(RTVIEvent.BotStartedSpeaking, () => {
      setAgentState('speaking', '正在回答...');
      moveSphereDown();
    });

    client.on(RTVIEvent.BotStoppedSpeaking, () => {
      setAgentState('idle', '请说话...');
      currentBotMsgElement = null;
      currentBotText = '';
    });

    // 4. Real-time Transcripts (Chat Bubbles in Scrollable Stream)
    client.on(RTVIEvent.UserTranscript, (data) => {
      if (data?.text && data.final) {
        appendChatMessage('user', data.text.trim());
      }
    });

    client.on(RTVIEvent.BotTranscript, (data) => {
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

    // 5. Function Calling / Tools Notification
    client.on(RTVIEvent.LLMFunctionCall, (call) => {
      const fnName = call?.function_name || 'Tool';
      const isKB = fnName.includes('knowledge');
      setAgentState('thinking', isKB ? '检索专属知识库...' : '实时联网搜索...');

      const card = document.createElement('div');
      card.className = 'tool-event-card';
      card.innerHTML = `
        <div class="tool-badge">
          <span>${isKB ? '📚 Misuedi ARK 知识库' : '🌐 Parallel 实时搜索'}</span>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted);">
          查询关键词: <em>${call?.args?.query || fnName}</em>
        </div>
      `;
      toolActivityFeed.appendChild(card);
    });

    // 6. Metrics & Latency Telemetry
    client.on(RTVIEvent.Metrics, (metrics) => {
      if (metrics?.ttfb) {
        const ttfb = Math.round(metrics.ttfb * 1000);
        metricTts.textContent = `${ttfb} ms`;
      }
    });

    // Initiate WebRTC connection to backend offer endpoint
    const webrtcUrl = `${window.location.origin}/api/offer`;
    await client.connect({ webrtcUrl });

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
  isConnected = false;
  hasSphereMovedDown = false;
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
