// Offline, volumetric humanoid illustration. Raw joints are never changed.
// All projection and mesh construction use one image-XYZ -> Y-up transform.
const add3 = (a, b) => a.map((v, i) => v + b[i]);
const sub3 = (a, b) => a.map((v, i) => v - b[i]);
const mul3 = (a, s) => a.map(v => v * s);
const dot3 = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const cross3 = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
const norm3 = a => Math.hypot(...a);
const unit3 = (a, fallback = [0, 1, 0]) => norm3(a) > 1e-6 ? mul3(a, 1 / norm3(a)) : fallback;
const mix3 = (a, b, t) => add3(mul3(a, 1-t), mul3(b, t));
const valid3 = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);
const jointNames = D.joint_names;
const modeSelect = document.getElementById('bodyMode');
const focusSelect = document.getElementById('personFocus');
const labelsToggle = document.getElementById('jointLabels');
const perspectiveToggle = document.getElementById('perspective');
const maxPeople3D = Math.max(0, ...D.frames_3d.map(people => people.length));
for (let index = 0; index < maxPeople3D; index++) {
  const option = document.createElement('option');
  option.value = String(index); option.textContent = 'ROMP ' + index;
  focusSelect.appendChild(option);
}
let camYaw = 0, camPitch = 0, zoom3D = 1, isDragging = false;
let dragStartX = 0, dragStartY = 0;
let bounds3D, center3D, baseScale3D, floorWorldY;
let followRoot = [0,0,0];
let effectiveFocus = 'all';
const boundsCache = new Map(), meshCache = new Map();
let smoothRenderer;
try { smoothRenderer = new SmoothBodyRenderer(canvas3d); }
catch(error) { console.warn('Smooth renderer unavailable', error); }
document.getElementById('renderStatus').textContent=smoothRenderer?.gl?'3D · ánh sáng mượt':'3D · chế độ tương thích';
const toWorld = p => [p[0]-followRoot[0]-center3D[0], center3D[1]+followRoot[1]-p[1], p[2]-followRoot[2]-center3D[2]];

function visiblePeople(frameIndex) {
  const people = D.frames_3d[frameIndex] || [];
  const selected = D.selected_indices_3d[frameIndex];
  if (focusSelect.value === 'target' && selected >= 0 && people[selected]) return [[selected, people[selected]]];
  if (focusSelect.value !== 'all' && focusSelect.value !== 'target') {
    const index = Number(focusSelect.value);
    if (people[index]) return [[index, people[index]]];
  }
  return people.map((person, index) => [index, person]);
}

function fit3D(kind=focusSelect.value) {
  let lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  // Cache fixed clip bounds separately for scene and focus views.
  if (boundsCache.has(kind)) [lo,hi]=boundsCache.get(kind);
  else {
  D.frames_3d.forEach((people, t) => {
    let selected = people;
    if (kind === 'target') selected = D.selected_indices_3d[t] >= 0 ? [people[D.selected_indices_3d[t]]] : [];
    else if (kind !== 'all') selected = [people[Number(kind)]];
    selected.filter(Boolean).forEach(person => {
      const root=kind!=='all' && valid3(person[0]) ? person[0] : [0,0,0];
      person.filter(valid3).forEach(point => {
        const p=sub3(point,root);
        for (let axis=0; axis<3; axis++) { lo[axis]=Math.min(lo[axis],p[axis]); hi[axis]=Math.max(hi[axis],p[axis]); }
      });
    });
  });
  if (!Number.isFinite(lo[0])) { lo = [0,0,0]; hi = [400,400,400]; }
  boundsCache.set(kind,[lo,hi]);
  }
  effectiveFocus=kind;
  bounds3D = [lo, hi]; center3D = mix3(lo, hi, 0.5);
  // Reference plane is below the lowest source point, never at a guessed Y.
  floorWorldY = center3D[1] - hi[1] - Math.max(6, (hi[1]-lo[1])*0.06);
  updateViewScale();
}

function updateViewScale() {
  const [lo,hi]=bounds3D, worldCorners = [];
  for (const x of [lo[0],hi[0]]) for (const y of [lo[1],hi[1]]) for (const z of [lo[2],hi[2]]) worldCorners.push(rotate3D([x-center3D[0],center3D[1]-y,z-center3D[2]]));
  const xs = worldCorners.map(p=>p[0]), ys=worldCorners.map(p=>p[1]);
  const padding=0.18*Math.max(...sub3(hi,lo),60);
  baseScale3D = Math.min((canvas3d.width-80)/Math.max(60,Math.max(...xs)-Math.min(...xs)+padding),
                        (canvas3d.height-100)/Math.max(60,Math.max(...ys)-Math.min(...ys)+padding));
}

function rotate3D(p) {
  const cy=Math.cos(camYaw), sy=Math.sin(camYaw), cp=Math.cos(camPitch), sp=Math.sin(camPitch);
  const x=p[0]*cy-p[2]*sy, z=p[0]*sy+p[2]*cy;
  // Positive pitch means camera above the body, looking downward.
  return [x, p[1]*cp+z*sp, -p[1]*sp+z*cp];
}
function projectWorld(p) {
  const q=rotate3D(p), extent=Math.max(...sub3(bounds3D[1],bounds3D[0]),100);
  const distance=extent*3, depth=distance+q[2];
  if (depth<=1) return null;
  const perspective=perspectiveToggle.checked ? distance/depth : 1;
  const scale=baseScale3D*zoom3D*perspective;
  return {x:canvas3d.width/2+q[0]*scale,y:canvas3d.height/2-q[1]*scale,depth:q[2],scale};
}
function project3D(p) { return valid3(p) ? projectWorld(toWorld(p)) : null; }

function rgb(hex) { return hex.match(/[0-9a-f]{2}/gi).map(v=>parseInt(v,16)); }
function shade(color, brightness) { return 'rgb('+rgb(color).map(v=>Math.round(Math.min(255,v*brightness))).join(',')+')'; }

function triangle(mesh,a,b,c,color,normals) {
  if (![a,b,c].every(valid3)) return;
  mesh.push({points:[a,b,c],color,normals});
}
function tube(mesh, a, b, r1, r2, color, steps=20) {
  if (!a || !b || norm3(sub3(b,a))<1e-5) return;
  const axis=unit3(sub3(b,a)), u=unit3(cross3(axis,Math.abs(axis[1])<0.9?[0,1,0]:[1,0,0]));
  const v=unit3(cross3(axis,u));
  const ring=(p,r,t)=>add3(p,add3(mul3(u,Math.cos(t)*r),mul3(v,Math.sin(t)*r)));
  for (let i=0;i<steps;i++) {
    const t=i/steps*Math.PI*2, next=(i+1)/steps*Math.PI*2;
    const p=ring(a,r1,t), q=ring(a,r1,next), s=ring(b,r2,t), w=ring(b,r2,next);
    const normal=t=>unit3(add3(add3(mul3(u,Math.cos(t)),mul3(v,Math.sin(t))),mul3(axis,(r1-r2)/norm3(sub3(b,a)))));
    const n=normal(t), nn=normal(next);
    triangle(mesh,p,s,q,color,[n,n,nn]); triangle(mesh,q,s,w,color,[nn,n,nn]);
    triangle(mesh,a,q,p,color,[mul3(axis,-1),mul3(axis,-1),mul3(axis,-1)]);
    triangle(mesh,b,s,w,color,[axis,axis,axis]);
  }
}
function ellipsoid(mesh, center, radii, color, basis=[[1,0,0],[0,1,0],[0,0,1]], rings=12, steps=24) {
  const point=(latitude,longitude)=>add3(center,add3(mul3(basis[0],radii[0]*Math.sin(latitude)*Math.cos(longitude)),
    add3(mul3(basis[1],radii[1]*Math.cos(latitude)),mul3(basis[2],radii[2]*Math.sin(latitude)*Math.sin(longitude)))));
  const normal=(lat,lon)=>unit3(add3(mul3(basis[0],Math.sin(lat)*Math.cos(lon)/radii[0]),
    add3(mul3(basis[1],Math.cos(lat)/radii[1]),mul3(basis[2],Math.sin(lat)*Math.sin(lon)/radii[2]))));
  const grid=Array.from({length:rings+1},(_,j)=>Array.from({length:steps},(_,i)=>{
    const a=j/rings*Math.PI,c=i/steps*2*Math.PI;
    return {point:point(a,c),normal:normal(a,c)};
  }));
  for (let j=0;j<rings;j++) for(let i=0;i<steps;i++) {
    const k=(i+1)%steps,a=grid[j][i],b=grid[j+1][i],c=grid[j][k],d=grid[j+1][k];
    triangle(mesh,a.point,b.point,c.point,color,[a.normal,b.normal,c.normal]);
    triangle(mesh,c.point,b.point,d.point,color,[c.normal,b.normal,d.normal]);
  }
}

function bodyMesh(person, color) {
  const p=person.map(point=>valid3(point)?toWorld(point):null), mesh=[];
  const good=p.filter(Boolean);
  if(good.length<2)return mesh;
  const length=(a,b)=>p[a]&&p[b]?norm3(sub3(p[a],p[b])):0;
  const shoulderWidth=length(16,17)||length(13,14)*2||length(1,2)*1.6;
  const torsoLength=length(12,0)||length(9,0)*1.5;
  const width=shoulderWidth>1e-4?Math.max(3,Math.min(shoulderWidth,torsoLength>1?torsoLength*.95:shoulderWidth)):0;
  if(!width)return mesh;
  const lateral=p[16]&&p[17]?unit3(sub3(p[16],p[17]),[1,0,0]):[1,0,0];
  const up=p[12]&&p[0]?unit3(sub3(p[12],p[0])):[0,1,0];
  const front=unit3(cross3(up,lateral),[0,0,-1]);
  const vertical=unit3(cross3(lateral,front));
  const basis=[lateral,vertical,front];
  const skin='#d5e2e8', trousers='#576d85', shoes='#34475c';
  const hipCenter=p[1]&&p[2]?mix3(p[1],p[2],0.5):p[0];
  const shoulderCenter=p[16]&&p[17]?mix3(p[16],p[17],0.5):p[9];
  const hipWidth=length(1,2)?Math.max(width*.5,length(1,2)):width*.6;
  if(hipCenter)ellipsoid(mesh,hipCenter,[hipWidth*.62,width*.22,width*.24],trousers,basis,10,24);
  const centers=[hipCenter,p[0],p[3],p[6],p[9],shoulderCenter].filter(Boolean);
  if(centers.length>=2) {
    const catmull=(a,b,c,d,t)=>a.map((_,i)=>.5*((2*b[i])+(-a[i]+c[i])*t+(2*a[i]-5*b[i]+4*c[i]-d[i])*t*t+(-a[i]+3*b[i]-3*c[i]+d[i])*t*t*t));
    const steps=24, segments=24;
    const ringCenters=Array.from({length:steps+1},(_,i)=>{
      const x=i/steps*(centers.length-1), k=Math.min(centers.length-2,Math.floor(x));
      return catmull(centers[Math.max(0,k-1)],centers[k],centers[k+1],centers[Math.min(centers.length-1,k+2)],x-k);
    });
    const rings=ringCenters.map((center,i)=>{
      const t=i/steps, half=(hipWidth*(1-t)+width*t)*.5*(1-.1*Math.sin(t*Math.PI));
      const depth=width*(.2+.035*Math.sin(t*Math.PI));
      return Array.from({length:segments},(_,j)=>{
        const a=j/segments*2*Math.PI;
        return {point:add3(center,add3(mul3(lateral,Math.cos(a)*half),mul3(front,Math.sin(a)*depth))),
                normal:unit3(add3(mul3(lateral,Math.cos(a)/half),mul3(front,Math.sin(a)/depth)))};
      });
    });
    for(let i=0;i<steps;i++)for(let j=0;j<segments;j++) {
      const k=(j+1)%segments, a=rings[i][j],b=rings[i+1][j],c=rings[i][k],d=rings[i+1][k];
      const material=i/steps<.17?trousers:color;
      triangle(mesh,a.point,b.point,c.point,material,[a.normal,b.normal,c.normal]);
      triangle(mesh,c.point,b.point,d.point,material,[c.normal,b.normal,d.normal]);
    }
    for(const i of [0,steps])for(let j=0;j<segments;j++) {
      const normal=mul3(up,i===0?-1:1);
      triangle(mesh,ringCenters[i],rings[i][j].point,rings[i][(j+1)%segments].point,i===0?trousers:color,[normal,normal,normal]);
    }
  }
  if(shoulderCenter&&p[12])tube(mesh,shoulderCenter,p[12],width*.13,width*.12,skin);
  for(const [a,b,r1,r2,material] of [[16,18,.13,.085,color],[18,20,.087,.06,skin],[17,19,.13,.085,color],[19,21,.087,.06,skin],
    [1,4,.185,.12,trousers],[4,7,.12,.073,trousers],[2,5,.185,.12,trousers],[5,8,.12,.073,trousers]]) {
    if(p[a]&&p[b])tube(mesh,p[a],p[b],width*r1,width*r2,material);
  }
  for(const [index,r,material] of [[16,.13,color],[17,.13,color],[18,.09,skin],[19,.09,skin],[4,.12,trousers],[5,.12,trousers],
    [7,.075,trousers],[8,.075,trousers],[20,.065,skin],[21,.065,skin]]) {
    if(p[index])ellipsoid(mesh,p[index],[width*r,width*r,width*r],material,basis,8,16);
  }
  for(const [wrist,hand] of [[20,22],[21,23]])if(p[wrist]&&p[hand]) {
    const axis=unit3(sub3(p[hand],p[wrist]),up), side=unit3(cross3(axis,front),lateral), normal=unit3(cross3(side,axis));
    ellipsoid(mesh,mix3(p[wrist],p[hand],.7),[width*.075,Math.max(width*.12,length(wrist,hand)*.75),width*.045],skin,[side,axis,normal],8,16);
  }
  for(const [ankle,toe] of [[7,10],[8,11]])if(p[ankle]&&p[toe]) {
    const axis=unit3(sub3(p[toe],p[ankle]),front), side=unit3(cross3(up,axis),lateral), normal=unit3(cross3(axis,side),up);
    ellipsoid(mesh,mix3(p[ankle],p[toe],.65),[width*.1,width*.065,Math.max(width*.17,length(ankle,toe)*.65)],shoes,[side,normal,axis],8,20);
  }
  if(p[15]&&p[12]) {
    const radius=width*.23, headUp=unit3(sub3(p[15],p[12]),up);
    const headCenter=add3(p[15],mul3(headUp,radius*.4));
    tube(mesh,p[12],headCenter,width*.095,width*.11,skin);
    ellipsoid(mesh,headCenter,[radius,radius*1.22,radius*.95],skin,basis,16,28);
    ellipsoid(mesh,add3(headCenter,mul3(front,radius*.94)),[radius*.13,radius*.2,radius*.18],skin,basis,8,16);
    for(const sign of [-1,1]) {
      const eye=add3(headCenter,add3(mul3(front,radius*.88),add3(mul3(lateral,sign*radius*.34),mul3(vertical,radius*.14))));
      ellipsoid(mesh,eye,[radius*.07,radius*.055,radius*.055],'#34475c',basis,6,12);
    }
  }
  return mesh;
}

function drawMeshCanvas(mesh) {
  const light=unit3([-0.4,0.8,-1]);
  const faces=[];
  for(const face of mesh) {
    const points=face.points.map(projectWorld);
    if(points.some(p=>!p)) continue;
    const normal=unit3(cross3(sub3(face.points[1],face.points[0]),sub3(face.points[2],face.points[0])));
    const viewNormal=rotate3D(normal);
    const brightness=0.52+0.48*Math.abs(dot3(viewNormal,light));
    faces.push({points,depth:points.reduce((s,p)=>s+p.depth,0)/3,color:shade(face.color,brightness)});
  }
  faces.sort((a,b)=>b.depth-a.depth);
  for(const face of faces) {
    ctx3d.beginPath();ctx3d.moveTo(face.points[0].x,face.points[0].y);
    ctx3d.lineTo(face.points[1].x,face.points[1].y);ctx3d.lineTo(face.points[2].x,face.points[2].y);ctx3d.closePath();
    ctx3d.fillStyle=face.color;ctx3d.fill();ctx3d.strokeStyle=face.color;ctx3d.lineWidth=.4;ctx3d.stroke();
  }
}

function drawReference() {
  const size=Math.max(...sub3(bounds3D[1],bounds3D[0]),100)*.7, step=size/5;
  ctx3d.lineWidth=1;ctx3d.strokeStyle='#243449';
  for(let i=-5;i<=5;i++) for(const endpoints of [
    [[i*step,floorWorldY,-size],[i*step,floorWorldY,size]],
    [[-size,floorWorldY,i*step],[size,floorWorldY,i*step]]]) {
    const [a,b]=endpoints.map(projectWorld);if(!a||!b)continue;
    ctx3d.beginPath();ctx3d.moveTo(a.x,a.y);ctx3d.lineTo(b.x,b.y);ctx3d.stroke();
  }
  // Compact orientation triad in a fixed corner.
  const origin=[canvas3d.width-60,canvas3d.height-50];
  for(const [axis,color,name] of [[[1,0,0],'#fb7185','X'],[[0,1,0],'#4ade80','Y ↑'],[[0,0,1],'#60a5fa','Z']]) {
    const q=rotate3D(axis);ctx3d.strokeStyle=color;ctx3d.fillStyle=color;
    ctx3d.beginPath();ctx3d.moveTo(...origin);ctx3d.lineTo(origin[0]+q[0]*28,origin[1]-q[1]*28);ctx3d.stroke();
    ctx3d.font='11px system-ui';ctx3d.fillText(name,origin[0]+q[0]*35,origin[1]-q[1]*35);
  }
}

function render3D(frameIndex) {
  const selected=D.selected_indices_3d[frameIndex], match=D.person_matches[frameIndex];
  const people=visiblePeople(frameIndex), mesh=[];
  const wanted=focusSelect.value;
  const present=wanted==='target'?selected>=0: wanted==='all'||Boolean(D.frames_3d[frameIndex]?.[Number(wanted)]);
  const scene=present?wanted:'all';
  if(effectiveFocus!==scene)fit3D(scene);
  updateViewScale();
  // Camera follows the pelvis in focus mode, leaving source coordinates intact.
  followRoot=scene!=='all' && people.length && valid3(people[0][1][0]) ? people[0][1][0] : [0,0,0];
  ctx3d.clearRect(0,0,canvas3d.width,canvas3d.height);drawReference();
  const colorFor=index=>index===selected?'#dfa153':PALETTE_3D[index%PALETTE_3D.length];
  const key=frameIndex+'|'+scene;
  let cached=meshCache.get(key);
  if(!cached){
    const empty=[];
    people.forEach(([index,person])=>{
      const actor=bodyMesh(person,colorFor(index));mesh.push(...actor);
      if(!actor.length)empty.push(index);
    });
    cached={mesh,buffer:smoothRenderer?.gl?smoothRenderer.pack(mesh):null,empty};
    meshCache.set(key,cached);if(meshCache.size>8)meshCache.delete(meshCache.keys().next().value);
  }
  if(smoothRenderer?.gl) {
    const buffer=modeSelect.value==='skeleton'?new Float32Array(0):cached.buffer;
    if(!smoothRenderer.draw(buffer,camYaw,camPitch,baseScale3D*zoom3D,Math.max(...sub3(bounds3D[1],bounds3D[0]),100),perspectiveToggle.checked))drawMeshCanvas(cached.mesh);
  } else if(modeSelect.value!=='skeleton')drawMeshCanvas(cached.mesh);
  people.forEach(([index,person])=>{
    const projected=person.map(project3D), color=colorFor(index);
    if(modeSelect.value!=='body'||cached.empty.includes(index)) {
      ctx3d.strokeStyle=modeSelect.value==='overlay'?'#f8fafc':color;ctx3d.fillStyle=ctx3d.strokeStyle;ctx3d.lineWidth=2;
      for(const [a,b] of BONES_3D) if(projected[a]&&projected[b]) {
        ctx3d.beginPath();ctx3d.moveTo(projected[a].x,projected[a].y);ctx3d.lineTo(projected[b].x,projected[b].y);ctx3d.stroke();
      }
      projected.forEach(p=>{if(p){ctx3d.beginPath();ctx3d.arc(p.x,p.y,2.5,0,Math.PI*2);ctx3d.fill();}});
    }
    if(labelsToggle.checked) projected.forEach((p,j)=>{if(p){ctx3d.font='10px system-ui';ctx3d.fillStyle='#f8fafc';ctx3d.fillText(j+': '+jointNames[j],p.x+5,p.y-3);}});
    const head=projected[15]||projected[12];
    if(head){ctx3d.font='12px system-ui';ctx3d.fillStyle=color;ctx3d.fillText('ROMP '+index+(index===selected?' · GHÉP 2D':''),head.x-25,head.y-25);}
  });
  let status=match.status==='matched'?'Đã ghép 2D → ROMP '+selected:match.status==='ambiguous'?'Chưa chắc người tương ứng · hiện các ứng viên nguồn':'Chưa ghép người · hiện các ứng viên nguồn';
  if(!present && wanted!=='target')status='ROMP '+wanted+' vắng tại frame này · chuyển sang toàn cảnh';
  document.getElementById('mappingStatus').textContent=status;
  if(!people.length){ctx3d.font='14px system-ui';ctx3d.fillStyle='#94a3b8';ctx3d.fillText('Nguồn ROMP không có pose ở frame này. Xem đối chiếu 2D.',30,canvas3d.height/2);}
}

function setView(yaw,pitch) {camYaw=yaw;camPitch=pitch;zoom3D=1;fit3D();renderCurrentFrame();}
document.getElementById('btnViewFront').onclick=()=>setView(0,0);
document.getElementById('btnViewSide').onclick=()=>setView(Math.PI/2,0);
document.getElementById('btnViewTop').onclick=()=>setView(0,Math.PI/2);
document.getElementById('btnViewReset').onclick=()=>setView(0,0);
focusSelect.onchange=()=>{zoom3D=1;fit3D();renderCurrentFrame();};
modeSelect.onchange=labelsToggle.onchange=perspectiveToggle.onchange=()=>renderCurrentFrame();
canvas3d.onpointerdown=e=>{isDragging=true;dragStartX=e.clientX;dragStartY=e.clientY;canvas3d.setPointerCapture(e.pointerId);};
canvas3d.onpointermove=e=>{
  if(!isDragging)return;
  camYaw+=(e.clientX-dragStartX)*.008;camPitch=Math.max(-1.5,Math.min(1.5,camPitch+(e.clientY-dragStartY)*.008));
  dragStartX=e.clientX;dragStartY=e.clientY;renderCurrentFrame();
};
canvas3d.onpointerup=canvas3d.onpointercancel=()=>{isDragging=false;};
canvas3d.onwheel=e=>{e.preventDefault();zoom3D=Math.max(.25,Math.min(5,zoom3D*Math.exp(-e.deltaY*.001)));renderCurrentFrame();};
canvas3d.style.touchAction='none';
canvas3d.addEventListener('webglcontextlost',event=>{
  event.preventDefault();
  document.getElementById('renderStatus').textContent='3D · chế độ tương thích';
  renderCurrentFrame();
});
canvas3d.addEventListener('webglcontextrestored',()=>{
  smoothRenderer=new SmoothBodyRenderer(canvas3d);meshCache.clear();
  document.getElementById('renderStatus').textContent='3D · ánh sáng mượt';
  renderCurrentFrame();
});
function resizeStage() {
  const stage=document.getElementById('stage3d'), ratio=Math.min(window.devicePixelRatio||1,2);
  const width=Math.round(stage.clientWidth*ratio), height=Math.round(stage.clientHeight*ratio);
  if(canvas3d.width!==width||canvas3d.height!==height) {
    canvas3d.width=canvas3dOverlay.width=width;canvas3d.height=canvas3dOverlay.height=height;
    meshCache.clear();fit3D(effectiveFocus);renderCurrentFrame();
  }
}
fit3D();
new ResizeObserver(resizeStage).observe(document.getElementById('stage3d'));
