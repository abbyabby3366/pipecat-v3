/**
 * WebGL Aurora Mesh Background (ReactBits style)
 * Powered by OGL with custom fluid wavy curtain shaders.
 */

import { Renderer, Program, Mesh, Triangle, Color } from 'ogl';

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
  uniform vec3 uColorStops[3];
  uniform float uAmplitude;

  // 2D Simplex Noise
  vec3 permute(vec3 x) { return mod(((x*34.0)+1.0)*x, 289.0); }

  float snoise(vec2 v){
    const vec4 C = vec4(0.211324865405187, 0.366025403784439,
             -0.577350269189626, 0.024390243902439);
    vec2 i  = floor(v + dot(v, C.yy) );
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod(i, 289.0);
    vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));
    vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy),
      dot(x12.zw,x12.zw)), 0.0);
    m = m*m ;
    m = m*m ;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
  }

  void main() {
    vec2 uv = vUv;
    float aspect = uResolution.x / uResolution.y;
    vec2 p = vec2(uv.x * aspect, uv.y);

    // Multi-frequency undulating wave
    float noise1 = snoise(vec2(p.x * 1.2 + uTime * 0.15, p.y * 0.8));
    float noise2 = snoise(vec2(p.x * 2.5 - uTime * 0.2, p.y * 1.5 + noise1 * 0.5));
    float wave = (noise1 * 0.6 + noise2 * 0.4) * uAmplitude;

    float gradientY = uv.y + wave * 0.25;

    // 3-color stop aurora blend
    vec3 color1 = uColorStops[0];
    vec3 color2 = uColorStops[1];
    vec3 color3 = uColorStops[2];

    vec3 col = mix(color1, color2, smoothstep(0.0, 0.5, gradientY));
    col = mix(col, color3, smoothstep(0.5, 1.0, gradientY));

    // Dark atmospheric vignette
    float vignette = 1.0 - length((uv - 0.5) * 1.2);
    vignette = clamp(vignette, 0.2, 1.0);

    // Base background blend
    vec3 bg = vec3(0.03, 0.035, 0.05);
    vec3 finalCol = mix(bg, col, 0.35 * vignette);

    gl_FragColor = vec4(finalCol, 1.0);
  }
`;

export class AuroraBackground {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.time = 0;
    this.running = true;

    // ReactBits Aurora color stops (#3A29FF, #FF94B4, #06B6D4)
    this.colorStops = [
      new Color('#3A29FF'), // Electric Indigo
      new Color('#A855F7'), // Purple
      new Color('#06B6D4'), // Cyan
    ];

    this.initWebGL();
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  initWebGL() {
    try {
      this.renderer = new Renderer({
        canvas: this.canvas,
        dpr: Math.min(window.devicePixelRatio, 1.5),
        alpha: false,
      });

      this.gl = this.renderer.gl;
      const geometry = new Triangle(this.gl);

      this.program = new Program(this.gl, {
        vertex: VERT_SHADER,
        fragment: FRAG_SHADER,
        uniforms: {
          uTime: { value: 0 },
          uResolution: { value: [window.innerWidth, window.innerHeight] },
          uAmplitude: { value: 1.0 },
          uColorStops: {
            value: [
              this.colorStops[0],
              this.colorStops[1],
              this.colorStops[2],
            ],
          },
        },
      });

      this.mesh = new Mesh(this.gl, { geometry, program: this.program });
      this.animate();
    } catch (e) {
      console.warn('Aurora WebGL init failed:', e);
    }
  }

  resize() {
    if (!this.renderer) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    this.renderer.setSize(w, h);
    if (this.program) {
      this.program.uniforms.uResolution.value = [w, h];
    }
  }

  animate() {
    if (!this.running || !this.renderer) return;

    this.time += 0.008;
    if (this.program) {
      this.program.uniforms.uTime.value = this.time;
      this.renderer.render({ scene: this.mesh });
    }

    requestAnimationFrame(() => this.animate());
  }

  destroy() {
    this.running = false;
  }
}
