"""Exercise the real viewer JS against missing-person and camera regressions."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_missing_association_and_partial_pose_remain_visible(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the offline viewer regression check")
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    # Anatomical SMPL24 fixture; Y follows image coordinates.
    pose = [[0, 0, 0], [12, 10, 0], [-12, 10, 0], [0, -15, 0],
            [12, 50, 0], [-12, 50, 0], [0, -30, 0], [12, 90, 0], [-12, 90, 0],
            [0, -40, 0], [12, 95, -10], [-12, 95, -10], [0, -60, 0],
            [10, -45, 0], [-10, -45, 0], [0, -75, 0], [25, -45, 0], [-25, -45, 0],
            [32, -10, 0], [-32, -10, 0], [35, 20, 0], [-35, 20, 0], [35, 28, 0], [-35, 28, 0]]
    payload = {"frames_3d": [[pose], [pose]], "selected_indices_3d": [0, -1],
               "person_matches": [{"status": "matched"}, {"status": "unmatched"}],
               "joint_names": [str(i) for i in range(24)]}
    runner = tmp_path / "check-viewer.cjs"
    runner.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
const document={getElementById(id){if(!elements.has(id))elements.set(id,{value:id==='personFocus'?'all':id==='bodyMode'?'body':'',checked:false,textContent:'',appendChild(){},clientWidth:800,clientHeight:560});return elements.get(id);},createElement(){return {};}};
const canvas={width:800,height:560,style:{},getContext(){return null;},addEventListener(){},setPointerCapture(){}};
const ctx=new Proxy({}, {get:(target,key)=>target[key]??(()=>{}),set:(target,key,value)=>{target[key]=value;return true;}});
const context=vm.createContext({document,canvas3d:canvas,canvas3dOverlay:{},ctx3d:ctx,
 D:JSON.parse(process.argv[2]),PALETTE_3D:['#38bdf8'],BONES_3D:[[0,1],[0,2]],
 window:{devicePixelRatio:1},ResizeObserver:class{observe(){}},console,renderCurrentFrame(){}});
for(const file of process.argv.slice(3))vm.runInContext(fs.readFileSync(file,'utf8'),context);
vm.runInContext(`
 assert(visiblePeople(1).length===1,'default view blank despite source pose');
 focusSelect.value='target';render3D(1);
 assert(visiblePeople(1).length===1 && effectiveFocus==='all','unmatched target hidden');
 focusSelect.value='99';render3D(1);
 assert(visiblePeople(1).length===1 && effectiveFocus==='all','absent raw candidate hidden');
 focusSelect.value='all';render3D(0);
 const person=D.frames_3d[0][0], partial=person.map(p=>p.slice());partial[16]=null;
 assert(bodyMesh(partial,'#38bdf8').length>0,'one missing shoulder discarded complete body');
 for(const yaw of [0,Math.PI/2,Math.PI])for(const pitch of [0,Math.PI/2,-.8]){
   camYaw=yaw;camPitch=pitch;updateViewScale();
   for(const face of bodyMesh(person,'#38bdf8'))for(const vertex of face.points){
     const point=projectWorld(vertex);assert(point&&Number.isFinite(point.x)&&Number.isFinite(point.y),'invalid projection');
     assert(point.x>=0&&point.x<=canvas3d.width&&point.y>=0&&point.y<=canvas3d.height,'mesh clipped after orbit');
   }
 }
 assert(bodyMesh(Array.from({length:24},()=>[0,0,0]),'#38bdf8').length===0,'fabricated body from collapsed pose');
 D.frames_3d[1]=[];render3D(1);assert(visiblePeople(1).length===0,'fabricated pose for empty source');
`,vm.createContext(Object.assign(context,{assert})));
console.log('visibility, partial-pose and orbit checks passed');
''', encoding="utf-8")
    result = subprocess.run([node, str(runner), json.dumps(payload),
                             str(scripts / "tracking_review_3d_webgl.js"),
                             str(scripts / "tracking_review_3d_renderer.js")],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
