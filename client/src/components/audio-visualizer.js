/**
 * Web Audio Visualizer Analyzer
 * Analyzes audio energy and frequency spectrum from microphone or speaker stream.
 */

export class AudioVisualizer {
  constructor(reactiveOrb) {
    this.orb = reactiveOrb;
    this.audioContext = null;
    this.analyser = null;
    this.source = null;
    this.freqData = null;
    this.running = false;
    this.activeSourceType = 'none'; // 'mic' | 'speaker' | 'none'
  }

  async initAudioContext() {
    if (!this.audioContext) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.audioContext = new AudioCtx();
    }
    if (this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
  }

  async attachStream(stream, sourceType = 'mic') {
    try {
      await this.initAudioContext();

      if (this.source) {
        try { this.source.disconnect(); } catch (_) {}
      }

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 128;
      this.analyser.smoothingTimeConstant = 0.8;

      this.source = this.audioContext.createMediaStreamSource(stream);
      this.source.connect(this.analyser);

      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.activeSourceType = sourceType;
      this.running = true;
      this.startLoop();
    } catch (e) {
      console.warn('AudioVisualizer attachStream error:', e);
    }
  }

  async attachAudioElement(audioElement) {
    try {
      await this.initAudioContext();

      if (this.source) {
        try { this.source.disconnect(); } catch (_) {}
      }

      this.analyser = this.audioContext.createAnalyser();
      this.analyser.fftSize = 128;
      this.analyser.smoothingTimeConstant = 0.8;

      this.source = this.audioContext.createMediaElementSource(audioElement);
      this.source.connect(this.analyser);
      this.analyser.connect(this.audioContext.destination);

      this.freqData = new Uint8Array(this.analyser.frequencyBinCount);
      this.activeSourceType = 'speaker';
      this.running = true;
      this.startLoop();
    } catch (e) {
      console.warn('AudioVisualizer attachAudioElement error:', e);
    }
  }

  startLoop() {
    const loop = () => {
      if (!this.running || !this.analyser) return;

      this.analyser.getByteFrequencyData(this.freqData);

      // Compute average RMS volume
      let sum = 0;
      for (let i = 0; i < this.freqData.length; i++) {
        sum += this.freqData[i];
      }
      const avg = sum / this.freqData.length;
      const normalizedEnergy = Math.min(1, avg / 128);

      this.orb.setAudioData(normalizedEnergy, this.freqData);

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }

  detach() {
    this.running = false;
    this.activeSourceType = 'none';
    if (this.source) {
      try { this.source.disconnect(); } catch (_) {}
      this.source = null;
    }
    this.orb.setAudioData(0, new Uint8Array(64));
  }
}
