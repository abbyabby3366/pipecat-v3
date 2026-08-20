# Frontend Integration Guide: Pipecat Voice Agent (Daily WebRTC + RTVI)

This document contains everything a frontend engineer needs to integrate any web client (Vanilla JS, React, Vue, Next.js, etc.) with the backend Pipecat voice assistant.

---

## 1. Overview & Architecture

The voice assistant uses **Daily.co WebRTC** for real-time bidirectional audio streaming and the **RTVI (Real-Time Voice Interface)** protocol for handling event frames (transcripts, agent states, tool calls).

```
┌────────────────────────────────────────────────────────┐
│                   Frontend Client                      │
└───────┬────────────────────────────────────────┬───────┘
        │ 1. POST /api/connect                   │ 3. WebRTC Audio & RTVI Events
        ▼                                        ▼
┌──────────────────┐               ┌───────────────────────────┐
│  FastAPI Backend │               │    Daily.co WebRTC Room   │
│  (Port 7860)     ├──────────────►│    (Audio + DataChannel)  │
└──────────────────┘ 2. Spawns Bot └───────────────────────────┘
```

1. **Handshake**: The frontend calls `POST /api/connect` on the backend.
2. **Room Creation**: The backend creates an ephemeral Daily.co WebRTC room and returns `{ room_url, token }`.
3. **Session Join**: The frontend initializes `PipecatClient` with `DailyTransport` and connects to the room.
4. **Real-time Interaction**:
   - Audio is streamed bidirectionally via WebRTC.
   - Text transcripts, speaking states, and tool/search operations stream live via **RTVI events**.

---

## 2. Package Installation

Install the required Pipecat and Daily SDKs:

```bash
npm install @pipecat-ai/client-js @pipecat-ai/daily-transport @daily-co/daily-js
```

---

## 3. Backend API Specification

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/connect` | `POST` | Creates an ephemeral Daily room (30 min expiry), starts the bot worker, and returns credentials. |
| `/api/health` | `GET` | Healthcheck endpoint (`{ status: "healthy", ... }`). |

### `POST /api/connect`
- **Request Headers**: `Content-Type: application/json`
- **Request Body**: None
- **Response**: `200 OK`
```json
{
  "room_url": "https://your-domain.daily.co/room-name",
  "token": "eyJhbGciOiJIUz..."
}
```

---

## 4. Complete Implementation (Vanilla JS / TypeScript)

```javascript
import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

// 1. Reference an <audio id="bot-audio" autoplay playsinline></audio> element in DOM
const botAudioElement = document.getElementById('bot-audio');

let client = null;

// ==========================================
// START CALL
// ==========================================
export async function startVoiceCall(backendBaseUrl = '') {
  // Step A: Request room credentials from backend
  const resp = await fetch(`${backendBaseUrl}/api/connect`, { method: 'POST' });
  if (!resp.ok) {
    const errorData = await resp.json().catch(() => ({}));
    throw new Error(errorData.detail || `Server error: ${resp.status}`);
  }
  const { room_url, token } = await resp.json();

  // Step B: Instantiate Pipecat Client
  client = new PipecatClient({
    transport: new DailyTransport(),
    enableMic: true,
    enableCam: false,
  });

  // Step C: Attach RTVI event listeners
  setupEventListeners(client);

  // Step D: Connect to the Daily room
  await client.connect({
    url: room_url,
    token: token,
  });
}

// ==========================================
// EVENT LISTENERS (RTVI)
// ==========================================
function setupEventListeners(client) {
  // 1. Audio Track: Attach incoming bot voice to <audio> tag
  client.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind === 'audio' && !participant?.local) {
      botAudioElement.srcObject = new MediaStream([track]);
      botAudioElement.play().catch(console.error);
    }
  });

  // 2. Lifecycle
  client.on(RTVIEvent.BotConnected, () => {
    console.log('Status: Bot is connected and ready.');
  });

  client.on(RTVIEvent.Disconnected, () => {
    console.log('Status: Disconnected.');
    if (botAudioElement.srcObject) botAudioElement.srcObject = null;
  });

  // 3. Speaking Turn Indicators
  client.on(RTVIEvent.UserStartedSpeaking, () => {
    console.log('State: [User Speaking] -> UI: Listening...');
  });

  client.on(RTVIEvent.UserStoppedSpeaking, () => {
    console.log('State: [User Stopped]  -> UI: Thinking...');
  });

  client.on(RTVIEvent.BotStartedSpeaking, () => {
    console.log('State: [Bot Speaking]   -> UI: Speaking...');
  });

  client.on(RTVIEvent.BotStoppedSpeaking, () => {
    console.log('State: [Bot Stopped]    -> UI: Idle');
  });

  // 4. Transcripts
  // User speech transcript (STT)
  client.on(RTVIEvent.UserTranscript, (data) => {
    if (data?.final && data.text?.trim()) {
      console.log(`User transcript: ${data.text.trim()}`);
      // Render user chat bubble
    }
  });

  // Streamed Bot response tokens (LLM token streaming)
  client.on(RTVIEvent.BotLlmText, (data) => {
    if (data?.text) {
      console.log(`Bot streaming token: ${data.text}`);
      // Append characters live to active bot chat bubble
    }
  });

  // Fallback for non-LLM bot transcripts (e.g., initial system greeting)
  client.on(RTVIEvent.BotTranscript, (data) => {
    if (data?.text) {
      console.log(`Bot system transcript: ${data.text}`);
    }
  });

  // 5. Tool & Function Calling (Knowledge Base & Web Search)
  client.on(RTVIEvent.LLMFunctionCall, (call) => {
    const fnName = call?.function_name || '';
    const query = call?.args?.query || call?.arguments?.query || '';
    const isKB = fnName.includes('knowledge');

    console.log(`Tool Started: ${isKB ? 'Knowledge Base' : 'Web Search'} for "${query}"`);
    // Show spinner/badge: "Searching knowledge base..." or "Searching the web..."
  });

  client.on(RTVIEvent.LLMFunctionCallStopped, (data) => {
    console.log('Tool Completed. Results:', data?.result);
    // data.result contains { records: [...] } or { results: [...] }
  });

  // 6. Error Handling
  client.on(RTVIEvent.Error, (err) => console.error('RTVI Error:', err));
  client.on(RTVIEvent.DeviceError, (err) => console.error('Device Error:', err));
}

// ==========================================
// CONTROLS
// ==========================================
export function toggleMute(isMuted) {
  if (client) {
    client.enableMic(!isMuted);
  }
}

export async function endVoiceCall() {
  if (client) {
    await client.disconnect();
    client = null;
  }
}
```

---

## 5. RTVI Event Reference Table

| Event (`RTVIEvent.<Name>`) | Payload | Action / UI Meaning |
| :--- | :--- | :--- |
| `TrackStarted` | `(track: MediaStreamTrack, participant)` | Incoming media track. Assign `new MediaStream([track])` to `<audio>` element to hear the bot. |
| `BotConnected` | None | Bot joined the room. Enable active in-call controls. |
| `Disconnected` | None | Session ended. Clean up audio elements and reset UI state. |
| `UserStartedSpeaking` | None | User is talking. Set status to **"Listening..."** / trigger user visualizer. |
| `UserStoppedSpeaking` | None | User finished turn. Set status to **"Thinking..."**. |
| `BotStartedSpeaking` | None | Bot began audio output. Set status to **"Speaking..."** / trigger bot visualizer. |
| `BotStoppedSpeaking` | None | Bot finished answering. Set status to **"Idle / Ready"**. |
| `UserTranscript` | `{ text: string, final: boolean }` | User STT transcript. Use `data.final === true` to finalize chat bubble. |
| `BotLlmText` | `{ text: string }` | Live token stream from LLM. Append characters to the bot message container. |
| `BotTranscript` | `{ text: string }` | Full transcript fallback (e.g. system notifications). |
| `LLMFunctionCall` / `LLMFunctionCallStarted` | `{ function_name: string, args: { query: string } }` | Fired when the agent invokes tools. Display searching indicator / query badge. |
| `LLMFunctionCallStopped` | `{ function_name: string, result: object }` | Tool execution finished. Contains retrieved search results or KB documents. |
| `Metrics` | `{ ttfb: number }` | Time-to-first-byte latency telemetry in seconds (e.g., `ttfb * 1000` ms). |
| `Error` / `DeviceError` | `Error object` | Handle mic permission errors or WebRTC network issues. |

---

## 6. Tool & Function Calling Data Structure

When tools are invoked, the backend emits `LLMFunctionCall` and `LLMFunctionCallStopped`:

### 1. ARK Knowledge Base (`search_knowledge_base`)
- **Fired on**: Questions mentioning ARK project / whitepaper / tokenomics / governance.
- **Finished Result (`data.result`)**:
```json
{
  "records": [
    {
      "document": "ARK_Whitepaper_v1.0.pdf",
      "score": 0.85,
      "content": "ARK provides an asynchronous consensus model..."
    }
  ]
}
```

### 2. Live Web Search (`search_web`)
- **Fired on**: Real-time news, today's weather, or volatile financial market queries.
- **Finished Result (`data.result`)**:
```json
{
  "results": [
    {
      "title": "Example Headline Title",
      "url": "https://example.com/news/123",
      "snippet": "Live summary excerpt text..."
    }
  ]
}
```

---

## 7. React / Next.js Hook Example

```tsx
import { useState, useRef, useCallback } from 'react';
import { PipecatClient, RTVIEvent } from '@pipecat-ai/client-js';
import { DailyTransport } from '@pipecat-ai/daily-transport';

export function usePipecatVoice(backendUrl = '') {
  const [isConnected, setIsConnected] = useState(false);
  const [agentState, setAgentState] = useState<'idle' | 'listening' | 'thinking' | 'speaking'>('idle');
  const [messages, setMessages] = useState<Array<{ role: 'user' | 'bot'; text: string }>>([]);
  const [isMuted, setIsMuted] = useState(false);

  const clientRef = useRef<PipecatClient | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const startCall = useCallback(async () => {
    // 1. Fetch credentials
    const res = await fetch(`${backendUrl}/api/connect`, { method: 'POST' });
    const { room_url, token } = await res.json();

    // 2. Initialize client
    const client = new PipecatClient({
      transport: new DailyTransport(),
      enableMic: true,
      enableCam: false,
    });
    clientRef.current = client;

    // 3. Audio track handler
    client.on(RTVIEvent.TrackStarted, (track, participant) => {
      if (track.kind === 'audio' && !participant?.local && audioRef.current) {
        audioRef.current.srcObject = new MediaStream([track]);
        audioRef.current.play().catch(console.error);
      }
    });

    // 4. Status updates
    client.on(RTVIEvent.BotConnected, () => setIsConnected(true));
    client.on(RTVIEvent.Disconnected, () => {
      setIsConnected(false);
      setAgentState('idle');
    });
    client.on(RTVIEvent.UserStartedSpeaking, () => setAgentState('listening'));
    client.on(RTVIEvent.UserStoppedSpeaking, () => setAgentState('thinking'));
    client.on(RTVIEvent.BotStartedSpeaking, () => setAgentState('speaking'));
    client.on(RTVIEvent.BotStoppedSpeaking, () => setAgentState('idle'));

    // 5. Transcripts
    client.on(RTVIEvent.UserTranscript, (data) => {
      if (data?.final && data.text?.trim()) {
        setMessages((prev) => [...prev, { role: 'user', text: data.text.trim() }]);
      }
    });

    client.on(RTVIEvent.BotLlmText, (data) => {
      if (data?.text) {
        setMessages((prev) => {
          const last = prev[prev.length - 1];
          if (last && last.role === 'bot') {
            return [...prev.slice(0, -1), { role: 'bot', text: last.text + data.text }];
          }
          return [...prev, { role: 'bot', text: data.text }];
        });
      }
    });

    // 6. Connect
    await client.connect({ url: room_url, token });
  }, [backendUrl]);

  const endCall = useCallback(async () => {
    if (clientRef.current) {
      await clientRef.current.disconnect();
      clientRef.current = null;
    }
    setIsConnected(false);
  }, []);

  const toggleMic = useCallback(() => {
    if (clientRef.current) {
      const nextMute = !isMuted;
      clientRef.current.enableMic(!nextMute);
      setIsMuted(nextMute);
    }
  }, [isMuted]);

  return {
    isConnected,
    agentState,
    messages,
    isMuted,
    audioRef,
    startCall,
    endCall,
    toggleMic,
  };
}
```

---

## 8. Crucial Frontend Checklist & Gotchas

1. **Hidden Audio Element**: Always include `<audio id="bot-audio" autoplay playsinline></audio>` in your DOM before `startCall()` is called. Without it, incoming bot audio will not play.
2. **User Gesture**: Microphone permissions must be triggered by a direct user gesture (e.g. clicking a "Start Call" button).
3. **CORS**: The backend FastAPI server has CORS enabled (`allow_origins=["*"]`), so local dev servers (`localhost:3000`, `localhost:5173`) can connect directly to `http://localhost:7860`.
4. **Screen Wake Lock (Optional Recommendation)**: For mobile web or prolonged voice sessions, use `navigator.wakeLock.request('screen')` during an active call to prevent the screen from sleeping.
