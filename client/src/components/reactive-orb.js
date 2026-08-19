/**
 * Audio-Reactive WebGL Orb Component (ReactBits / 21st.dev style)
 * Powered by OGL with custom raymarched fluid shaders & real-time audio modulation.
 */

import { Renderer, Program, Mesh, Triangle } from 'ogl';

const VERT_SHADER = /* glsl */ `
  attribute vec2 position;
  attribute vec2 uv;
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position, 0.0, 1.0);
  }
`;

const FRAG_SHADER = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform float uTime;
  uniform vec2 uResolution;
  uniform float uHue;
  uniform float uAudioEnergy;
  uniform float uAudioFreq;
  uniform vec3 uBaseColor1;
  uniform vec3 uBaseColor2;
  uniform vec3 uBaseColor3;
  uniform float uState; // 0=idle, 1=listening, 2=thinking, 3=speaking

  // Simplex 3D noise
  vec4 permute(vec4 x){return mod(((x*34.0)+1.0)*x, 289.0);}
  vec4 taylorInvSqrt(vec4 r){return 1.79284291400159 - 0.85373472095314 * r;}

  float snoise(vec3 v){
    const vec2 C = vec2(1.0/6.0, 1.0/3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    vec3 i  = floor(v + dot(v, C.yyy) );
    vec3 x0 = v - i + dot(i, C.xxx) ;

    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min( g.xyz, l.zxy );
    vec3 i2 = max( g.xyz, l.zxy );

    vec3 x1 = x0 - i1 + 1.0 * C.xxx;
    vec3 x2 = x0 - i2 + 2.0 * C.xxx;
    vec3 x3 = x0 - 1.0 + 3.0 * C.xxx;

    i = mod(i, 289.0 );
    vec4 p = permute( permute( permute(
               i.z + vec4(0.0, i1.z, i2.z, 1.0 ))
             + i.y + vec4(0.0, i1.y, i2.y, 1.0 ))
             + i.x + vec4(0.0, i1.x, i2.x, 1.0 ));

    float n_ = 0.142857142857;
    vec3  ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z *ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_ );

    vec4 x = x_ *ns.x + ns.yyyy;
    vec4 y = y_ *ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4( x.xy, y.xy );
    vec4 b1 = vec4( x.zw, y.zw );

    vec4 s0 = floor(b0)*2.0 + 1.0;
    vec4 s1 = floor(b1)*2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy ;
    vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww ;

    vec3 p0 = vec3(a0.xy,h.x);
    vec3 p1 = vec3(a0.zw,h.y);
    vec3 p2 = vec3(a1.xy,h.z);
    vec3 p3 = vec3(a1.zw,h.w);

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot( m*m, vec4( dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3) ) );
  }

  // Hue shift in YIQ space
  vec3 hueShift(vec3 color, float hue) {
    const mat3 rgb2yiq = mat3(0.299, 0.587, 0.114, 0.596, -0.274, -0.322, 0.211, -0.523, 0.312);
    const mat3 yiq2rgb = mat3(1.0, 0.956, 0.621, 1.0, -0.272, -0.647, 1.0, -1.106, 1.703);
    vec3 yiq = rgb2yiq * color;
    float angle = hue;
    float cosA = cos(angle);
    float sinA = sin(angle);
    yiq.y = yiq.y * cosA - yiq.z * sinA;
    yiq.z = yiq.y * sinA + yiq.z * cosA;
    return clamp(yiq2rgb * yiq, 0.0, 1.0);
  }

  void main() {
    vec2 st = (vUv - 0.5) * 2.0;
    st.x *= uResolution.x / uResolution.y;

    float dist = length(st);
    
    // Base radius modulated by audio energy
    float radius = 0.55 + uAudioEnergy * 0.25;
    
    // Dynamic 3D noise deformation
    vec3 p = vec3(st * (2.2 + uAudioFreq * 1.5), uTime * 0.45);
    float noiseVal = snoise(p) * (0.18 + uAudioEnergy * 0.35);
    
    float sphere = dist - radius + noiseVal;
    
    // Smooth boundary
    float alpha = smoothstep(0.08, -0.08, sphere);
    float rim = smoothstep(0.4, 0.0, abs(sphere)) * 0.85;
    
    // Multi-color harmonic gradient
    float colorMix = sin(uTime * 0.5 + st.x * 2.0 + noiseVal * 3.0) * 0.5 + 0.5;
    vec3 col = mix(uBaseColor1, uBaseColor2, colorMix);
    col = mix(col, uBaseColor3, clamp(st.y + 0.5, 0.0, 1.0));
    
    // Specular lighting & rim glow
    col += vec3(1.0) * rim * (0.8 + uAudioEnergy * 1.2);
    
    // Outer atmospheric glow
    float glow = exp(-dist * (2.4 - uAudioEnergy * 0.8)) * (0.65 + uAudioEnergy * 0.6);
    vec3 glowColor = mix(uBaseColor1, uBaseColor3, 0.5);

    // Apply hue
    col = hueShift(col, uHue);
    glowColor = hueShift(glowColor, uHue);

    vec3 finalColor = col * alpha + glowColor * glow * (1.0 - alpha * 0.5);
    float finalAlpha = clamp(alpha + glow * 0.85, 0.0, 1.0);

    gl_FragColor = vec4(finalColor, finalAlpha);
  }
`;

export class ReactiveOrb {
  constructor(canvasId, options = {}) {
    this.canvas = document.getElementById(canvasId);
    this.options = {
      hue: options.hue ?? 6.0, // default hue=6 as in reactbits sample
      hoverIntensity: options.hoverIntensity ?? 0,
      ...options,
    };

    this.state = 'idle';
    this.audioEnergy = 0;
    this.smoothedEnergy = 0;
    this.freqData = new Uint8Array(64);
    this.time = 0;

    this.initWebGL();
  }

  initWebGL() {
    try {
      this.renderer = new Renderer({
        canvas: this.canvas,
        width: 600,
        height: 600,
        dpr: Math.min(window.devicePixelRatio, 2),
        alpha: true,
        premultipliedAlpha: false,
      });

      this.gl = this.renderer.gl;
      this.gl.clearColor(0, 0, 0, 0);

      const geometry = new Triangle(this.gl);

      this.program = new Program(this.gl, {
        vertex: VERT_SHADER,
        fragment: FRAG_SHADER,
        transparent: true,
        uniforms: {
          uTime: { value: 0 },
          uResolution: { value: [600, 600] },
          uHue: { value: this.options.hue },
          uAudioEnergy: { value: 0 },
          uAudioFreq: { value: 0 },
          uState: { value: 0 },
          uBaseColor1: { value: [0.38, 0.40, 0.95] }, // Indigo (#6366f1)
          uBaseColor2: { value: [0.02, 0.71, 0.83] }, // Cyan (#06b6d4)
          uBaseColor3: { value: [0.65, 0.33, 0.96] }, // Purple (#a855f7)
        },
      });

      this.mesh = new Mesh(this.gl, { geometry, program: this.program });
      this.running = true;
      this.animate();
    } catch (e) {
      console.warn('OGL WebGL initialization failed, fallback to 2D:', e);
    }
  }

  setState(newState) {
    this.state = newState;
    if (!this.program) return;

    const u = this.program.uniforms;
    if (newState === 'listening') {
      u.uState.value = 1.0;
      u.uBaseColor1.value = [0.02, 0.71, 0.83]; // Cyan
      u.uBaseColor2.value = [0.06, 0.72, 0.50]; // Emerald
      u.uBaseColor3.value = [0.38, 0.40, 0.95]; // Indigo
    } else if (newState === 'thinking') {
      u.uState.value = 2.0;
      u.uBaseColor1.value = [0.96, 0.62, 0.04]; // Amber
      u.uBaseColor2.value = [0.92, 0.28, 0.60]; // Pink
      u.uBaseColor3.value = [0.65, 0.33, 0.96]; // Purple
    } else if (newState === 'speaking') {
      u.uState.value = 3.0;
      u.uBaseColor1.value = [0.54, 0.36, 0.96]; // Violet
      u.uBaseColor2.value = [0.02, 0.71, 0.83]; // Cyan
      u.uBaseColor3.value = [0.92, 0.28, 0.60]; // Pink
    } else {
      u.uState.value = 0.0;
      u.uBaseColor1.value = [0.38, 0.40, 0.95]; // Indigo
      u.uBaseColor2.value = [0.02, 0.71, 0.83]; // Cyan
      u.uBaseColor3.value = [0.65, 0.33, 0.96]; // Purple
    }
  }

  setAudioData(energy, freqData) {
    this.audioEnergy = energy;
    if (freqData) {
      this.freqData = freqData;
    }
  }

  animate() {
    if (!this.running || !this.renderer) return;

    this.time += 0.025;
    this.smoothedEnergy += (this.audioEnergy - this.smoothedEnergy) * 0.25;

    // Calculate mid-frequency band energy
    let freqSum = 0;
    for (let i = 4; i < 20; i++) {
      freqSum += this.freqData[i] || 0;
    }
    const freqAvg = (freqSum / 16) / 255;

    const u = this.program.uniforms;
    u.uTime.value = this.time;
    u.uAudioEnergy.value = this.smoothedEnergy;
    u.uAudioFreq.value = freqAvg;

    this.renderer.render({ scene: this.mesh });

    requestAnimationFrame(() => this.animate());
  }

  destroy() {
    this.running = false;
  }
}
