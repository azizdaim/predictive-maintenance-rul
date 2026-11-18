const THREE_URL = "__THREE_URL__";
let threeModule = null;
const instances = new WeakMap();

// What each section does. Sensor numbers come from Python (src/features/sensors.py)
const PARTS = {
  fan: ["Fan", "Pulls in air and drives most of the thrust."],
  lpc: ["Low-pressure compressor", "First squeeze of the core air."],
  hpc: ["High-pressure compressor", "Raises pressure for combustion."],
  combustor: ["Combustor", "Burns fuel with the compressed air."],
  hpt: ["High-pressure turbine", "Drives the compressor; the hottest part."],
  lpt: ["Low-pressure turbine", "Drives the fan."],
  nacelle: ["Nacelle", "Outer casing around the whole engine."],
};

function buildEngine(THREE, root, data) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  root.prepend(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(4.9, 2.3, 6.9);
  camera.lookAt(0, -0.1, 0);

  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x000000, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 2.2);
  key.position.set(5, 7, 6);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x3fd0ff, 1.6);
  rim.position.set(-7, 2, -5);
  scene.add(rim);
  const coreLight = new THREE.PointLight(0xffffff, 6, 4, 2);
  scene.add(coreLight);

  const materials = {};
  function material(part, options) {
    const m = new THREE.MeshStandardMaterial(options);
    m.userData.baseEmissive = m.emissive.clone();
    m.userData.baseIntensity = m.emissiveIntensity;
    (materials[part] = materials[part] || []).push(m);
    return m;
  }
  function tag(mesh, part) { mesh.userData.part = part; return mesh; }

  // Engine is built along the Y axis (how lathe shapes work), then laid flat
  const engine = new THREE.Group();
  const pivot = new THREE.Group();
  engine.rotation.z = -Math.PI / 2;
  pivot.add(engine);
  scene.add(pivot);

  const v = (r, y) => new THREE.Vector2(r, y);

  // Leave a wedge open that faces up and toward the viewer, to show the inside
  const CUT_START = Math.PI * 1.975;
  const CUT_LENGTH = Math.PI * 1.45;

  // Nacelle with a wedge cut away so the inside is visible
  const cowl = [
    v(1.26, 1.95), v(1.40, 1.86), v(1.48, 1.45), v(1.48, 0.35), v(1.36, -0.75),
    v(1.12, -1.25), v(1.04, -1.22), v(1.18, -0.55), v(1.25, 0.5), v(1.24, 1.55),
    v(1.22, 1.9), v(1.26, 1.95),
  ];
  const nacelle = tag(new THREE.Mesh(
    new THREE.LatheGeometry(cowl, 128, CUT_START, CUT_LENGTH),
    material("nacelle", { color: 0xdfe5ee, metalness: 0.55, roughness: 0.32, side: THREE.DoubleSide }),
  ), "nacelle");
  engine.add(nacelle);

  const metal = (part, color = 0xb8c2d0) =>
    material(part, { color, metalness: 0.9, roughness: 0.28 });

  // Spinner cone and fan
  const spinner = tag(new THREE.Mesh(
    new THREE.LatheGeometry([v(0, 2.32), v(0.16, 2.2), v(0.3, 1.98), v(0.36, 1.82), v(0, 1.82)], 48),
    material("fan", { color: 0xeef2f7, metalness: 0.7, roughness: 0.25 }),
  ), "fan");
  const fanRotor = new THREE.Group();
  fanRotor.add(spinner);
  const bladeGeometry = new THREE.BoxGeometry(0.86, 0.34, 0.035);
  const bladeMaterial = metal("fan", 0xc9d2de);
  for (let i = 0; i < 22; i++) {
    const arm = new THREE.Group();
    arm.rotation.y = (i / 22) * Math.PI * 2;
    const blade = tag(new THREE.Mesh(bladeGeometry, bladeMaterial), "fan");
    blade.position.set(0.78, 1.74, 0);
    blade.rotation.x = 0.65;
    arm.add(blade);
    fanRotor.add(arm);
  }
  engine.add(fanRotor);

  // A bladed disc: one ring of blades around a hub, as a single instanced mesh
  function stage(part, y, outer, count, mat) {
    const group = new THREE.Group();
    const hub = 0.26;
    const disc = tag(new THREE.Mesh(new THREE.CylinderGeometry(hub, hub, 0.05, 32), mat), part);
    disc.position.y = y;
    group.add(disc);
    const length = outer - hub;
    const blades = tag(new THREE.InstancedMesh(new THREE.BoxGeometry(length, 0.07, 0.02), mat, count), part);
    const dummy = new THREE.Object3D();
    for (let i = 0; i < count; i++) {
      const a = (i / count) * Math.PI * 2;
      dummy.position.set(Math.cos(a) * (hub + length / 2), y, Math.sin(a) * (hub + length / 2));
      dummy.rotation.set(0.5, -a, 0);
      dummy.updateMatrix();
      blades.setMatrixAt(i, dummy.matrix);
    }
    group.add(blades);
    return group;
  }

  const coreRotor = new THREE.Group();
  const lpcMat = metal("lpc", 0xaab6c6);
  [[1.45, 0.86], [1.3, 0.8], [1.15, 0.74]].forEach(([y, r]) => coreRotor.add(stage("lpc", y, r, 34, lpcMat)));
  const hpcMat = metal("hpc", 0x9aa7b8);
  [[0.95, 0.66], [0.82, 0.62], [0.69, 0.58], [0.56, 0.55], [0.43, 0.52]].forEach(([y, r]) =>
    coreRotor.add(stage("hpc", y, r, 40, hpcMat)));
  const turbineMat = material("hpt", { color: 0x8a7a6a, metalness: 0.8, roughness: 0.35, emissive: 0xff7a2f, emissiveIntensity: 0.35 });
  [[-0.32, 0.6], [-0.44, 0.63]].forEach(([y, r]) => coreRotor.add(stage("hpt", y, r, 44, turbineMat)));
  const lptMat = material("lpt", { color: 0x7d8796, metalness: 0.85, roughness: 0.3, emissive: 0xff7a2f, emissiveIntensity: 0.12 });
  [[-0.6, 0.7], [-0.73, 0.76], [-0.86, 0.82]].forEach(([y, r]) => coreRotor.add(stage("lpt", y, r, 46, lptMat)));
  const shaft = tag(new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 3.0, 24), metal("hpc", 0x6b7685)), "hpc");
  shaft.position.y = 0.35;
  coreRotor.add(shaft);
  engine.add(coreRotor);

  // Combustor ring, glowing like a flame
  const combustor = tag(new THREE.Mesh(
    new THREE.TorusGeometry(0.46, 0.09, 24, 96),
    material("combustor", { color: 0x3a1f10, emissive: 0xff8a3d, emissiveIntensity: 1.6, roughness: 0.6 }),
  ), "combustor");
  combustor.rotation.x = Math.PI / 2;
  combustor.position.y = 0.08;
  engine.add(combustor);
  coreLight.position.set(0.08, 0, 0);

  // Core casing and exhaust plug
  const casing = tag(new THREE.Mesh(
    new THREE.LatheGeometry([v(0.9, 1.6), v(0.72, 0.9), v(0.62, 0.3), v(0.7, -0.4), v(0.86, -1.0)], 96, CUT_START, CUT_LENGTH),
    material("hpc", { color: 0x2a3446, metalness: 0.6, roughness: 0.5, side: THREE.DoubleSide }),
  ), "hpc");
  engine.add(casing);
  const plug = tag(new THREE.Mesh(
    new THREE.LatheGeometry([v(0.42, -0.95), v(0.36, -1.35), v(0.16, -1.75), v(0, -1.86)], 48),
    metal("lpt", 0x9aa3b0),
  ), "lpt");
  engine.add(plug);

  // Status halo around the intake lip
  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(1.36, 0.018, 12, 160),
    new THREE.MeshBasicMaterial({ color: 0x3fd0ff, transparent: true, opacity: 0.9 }),
  );
  halo.rotation.x = Math.PI / 2;
  halo.position.y = 1.96;
  engine.add(halo);

  // Interaction: drag to rotate, gentle auto-rotation when idle
  const tooltip = root.querySelector(".tooltip");
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let dragging = false, lastX = 0, lastY = 0, idleSince = performance.now();
  let targetYaw = 0.12, yaw = 0.12, targetTilt = 0.16, tilt = 0.16;
  let hovered = null;

  function setHighlight(part) {
    if (hovered === part) return;
    for (const m of materials[hovered] || []) {
      m.emissive.copy(m.userData.baseEmissive);
      m.emissiveIntensity = m.userData.baseIntensity;
    }
    hovered = part;
    for (const m of materials[hovered] || []) {
      m.emissive.set(0x3fd0ff);
      m.emissiveIntensity = Math.max(m.userData.baseIntensity, 0.45);
    }
  }

  root.onpointerdown = (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; root.setPointerCapture(e.pointerId); };
  root.onpointerup = (e) => { dragging = false; idleSince = performance.now(); root.releasePointerCapture(e.pointerId); };
  root.onpointerleave = () => { setHighlight(null); tooltip.hidden = true; };
  root.onpointermove = (e) => {
    const rect = root.getBoundingClientRect();
    if (dragging) {
      targetYaw += (e.clientX - lastX) * 0.008;
      targetTilt = Math.max(-0.5, Math.min(0.6, targetTilt + (e.clientY - lastY) * 0.005));
      lastX = e.clientX; lastY = e.clientY;
      idleSince = performance.now();
      tooltip.hidden = true;
      return;
    }
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pivot.children, true).find((h) => h.object.userData.part);
    const part = hit ? hit.object.userData.part : null;
    setHighlight(part);
    if (part) {
      const [title, text] = PARTS[part];
      const sensors = state.sensors[part];
      const sensorText = sensors && sensors.length
        ? "<br>Sensor" + (sensors.length > 1 ? "s " : " ") + sensors.join(", ")
        : "";
      tooltip.innerHTML = "<b>" + title + "</b>" + text + sensorText;
      tooltip.hidden = false;
      tooltip.style.left = Math.min(e.clientX - rect.left + 14, rect.width - 250) + "px";
      tooltip.style.top = Math.max(e.clientY - rect.top - 10, 8) + "px";
    } else {
      tooltip.hidden = true;
    }
  };

  const state = { fanSpeed: 0.6, status: new THREE.Color(0x3fd0ff), glow: 0.5, sensors: {} };
  const chip = root.querySelector(".chip");

  function update(next) {
    root.style.height = (next.height || 440) + "px";
    state.sensors = next.sensors || {};
    state.fanSpeed = Math.max(0, Math.min(1, next.fan_speed ?? 0.6));
    state.status.set(next.status_color || "#3FD0FF");
    state.glow = Math.max(0, Math.min(1, next.glow ?? 0.5));
    halo.material.color.copy(state.status);
    coreLight.color.copy(state.status);
    for (const m of [turbineMat, lptMat]) {
      m.userData.baseEmissive = state.status.clone().lerp(new THREE.Color(0xff7a2f), 0.35);
      if (hovered !== "hpt" && hovered !== "lpt") m.emissive.copy(m.userData.baseEmissive);
    }
    // Screen readers get one sentence instead of a silent canvas
    root.setAttribute("role", "img");
    root.setAttribute("aria-label", next.description || "3D turbofan engine");
    chip.hidden = !next.label;
    chip.querySelector(".chip-text").textContent = next.label || "";
    chip.querySelector(".dot").style.color = next.status_color || "#3FD0FF";
    chip.querySelector(".dot").style.background = next.status_color || "#3FD0FF";
  }

  function resize() {
    const w = root.clientWidth, h = root.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(root);

  let frame = 0;
  const clock = new THREE.Clock();
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  function tick() {
    frame = requestAnimationFrame(tick);
    const dt = Math.min(clock.getDelta(), 0.05);
    const t = clock.elapsedTime;
    // Visitors who ask their device to reduce motion get a still engine
    const still = reducedMotion.matches;
    if (!still && !dragging && performance.now() - idleSince > 2500) {
      targetYaw += dt * 0.18;
    }
    yaw += (targetYaw - yaw) * 0.08;
    tilt += (targetTilt - tilt) * 0.08;
    pivot.rotation.set(tilt, yaw, 0);
    if (!still) {
      fanRotor.rotation.y += dt * (2 + state.fanSpeed * 14);
      coreRotor.rotation.y += dt * (4 + state.fanSpeed * 22);
    }
    const pulse = still ? 1 : 0.75 + 0.25 * Math.sin(t * 3);
    coreLight.intensity = 3 + state.glow * 9 * pulse;
    halo.material.opacity = 0.55 + 0.4 * pulse;
    renderer.render(scene, camera);
  }

  update(data);
  resize();
  tick();

  function dispose() {
    cancelAnimationFrame(frame);
    observer.disconnect();
    root.onpointerdown = root.onpointerup = root.onpointermove = root.onpointerleave = null;
    scene.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) [].concat(o.material).forEach((m) => m.dispose());
    });
    renderer.dispose();
    renderer.domElement.remove();
  }
  return { update, dispose };
}

export default function (component) {
  const { data, parentElement } = component;
  const root = parentElement.querySelector(".engine-root");
  let instance = instances.get(parentElement);

  if (!instance) {
    instance = { engine: null, pending: data, disposed: false };
    instances.set(parentElement, instance);
    root.style.height = ((data && data.height) || 440) + "px";
    const start = (THREE) => {
      threeModule = THREE;
      if (!instance.disposed) instance.engine = buildEngine(THREE, root, instance.pending);
    };
    if (threeModule) {
      start(threeModule);
    } else {
      import(THREE_URL).then(start).catch(() => {
        root.querySelector(".fallback").hidden = false;
      });
    }
  } else if (instance.engine) {
    instance.engine.update(data);
  } else {
    instance.pending = data;
  }

  return () => {
    instance.disposed = true;
    if (instance.engine) instance.engine.dispose();
    instances.delete(parentElement);
  };
}
