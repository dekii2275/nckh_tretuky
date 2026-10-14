// Smooth, depth-tested rendering without network dependencies.
class SmoothBodyRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.gl = canvas.getContext('webgl', {antialias:true, alpha:true, preserveDrawingBuffer:true});
    if (!this.gl) return;
    const gl=this.gl;
    const vertex=`
      attribute vec3 position; attribute vec3 normal; attribute vec3 color;
      uniform vec4 angles; uniform vec4 viewport; uniform float perspective;
      varying vec3 vNormal; varying vec3 vColor; varying vec3 vPosition;
      vec3 rotate(vec3 p) {
        float x=p.x*angles.x-p.z*angles.y, z=p.x*angles.y+p.z*angles.x;
        return vec3(x,p.y*angles.z+z*angles.w,-p.y*angles.w+z*angles.z);
      }
      void main() {
        vec3 q=rotate(position); float distance=viewport.w*3.0;
        float w=mix(1.0,max(1.0,distance+q.z),perspective);
        float f=mix(1.0,distance,perspective);
        gl_Position=vec4(q.x*viewport.z*2.0/viewport.x*f,
                         q.y*viewport.z*2.0/viewport.y*f,
                         clamp(q.z/(viewport.w*4.0),-0.99,0.99)*w,w);
        vNormal=rotate(normal); vColor=color; vPosition=q/viewport.w;
      }`;
    const fragment=`
      precision mediump float;
      varying vec3 vNormal; varying vec3 vColor; varying vec3 vPosition;
      void main() {
        vec3 N=normalize(vNormal);
        if(N.z>0.0)N=-N;
        vec3 key=normalize(vec3(-0.45,0.85,-1.0));
        vec3 fill=normalize(vec3(0.8,0.2,-0.55));
        float diffuse=max(0.0,dot(N,key)), fillLight=max(0.0,dot(N,fill));
        float hemisphere=0.5+0.5*N.y;
        float specular=pow(max(0.0,dot(N,normalize(key+vec3(0.0,0.0,-1.0)))),36.0);
        float rim=pow(1.0-abs(N.z),3.0);
        vec3 lit=vColor*(0.34+0.45*diffuse+0.13*fillLight+0.12*hemisphere)
                 +vec3(0.19)*specular+vec3(0.05,0.08,0.1)*rim;
        gl_FragColor=vec4(lit,1.0);
      }`;
    const compile=(kind,source)=>{
      const shader=gl.createShader(kind);gl.shaderSource(shader,source);gl.compileShader(shader);
      if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    this.program=gl.createProgram();
    gl.attachShader(this.program,compile(gl.VERTEX_SHADER,vertex));
    gl.attachShader(this.program,compile(gl.FRAGMENT_SHADER,fragment));gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));
    this.buffer=gl.createBuffer();this.locations={};
    for(const name of ['angles','viewport','perspective'])this.locations[name]=gl.getUniformLocation(this.program,name);
    this.attributes=['position','normal','color'].map(name=>gl.getAttribLocation(this.program,name));
    gl.enable(gl.DEPTH_TEST);gl.depthFunc(gl.LEQUAL);gl.disable(gl.CULL_FACE);
    gl.clearColor(0,0,0,0);
  }
  pack(mesh) {
    const buffer=new Float32Array(mesh.length*27);
    const colors=new Map();let offset=0;
    for(const face of mesh) {
      let color=colors.get(face.color);
      if(!color){color=rgb(face.color).map(v=>v/255);colors.set(face.color,color);}
      const fallback=unit3(cross3(sub3(face.points[1],face.points[0]),sub3(face.points[2],face.points[0])));
      for(let vertex=0;vertex<3;vertex++) {
        buffer.set(face.points[vertex],offset);offset+=3;
        buffer.set(face.normals?face.normals[vertex]:fallback,offset);offset+=3;
        buffer.set(color,offset);offset+=3;
      }
    }
    return buffer;
  }
  draw(buffer, yaw, pitch, scale, extent, perspective) {
    const gl=this.gl;if(!gl || gl.isContextLost())return false;
    gl.viewport(0,0,this.canvas.width,this.canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.useProgram(this.program);
    gl.uniform4f(this.locations.angles,Math.cos(yaw),Math.sin(yaw),Math.cos(pitch),Math.sin(pitch));
    gl.uniform4f(this.locations.viewport,this.canvas.width,this.canvas.height,scale,extent);
    gl.uniform1f(this.locations.perspective,perspective?1:0);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,buffer,gl.DYNAMIC_DRAW);
    this.attributes.forEach((attribute,i)=>{
      gl.enableVertexAttribArray(attribute);gl.vertexAttribPointer(attribute,3,gl.FLOAT,false,36,i*12);
    });
    gl.drawArrays(gl.TRIANGLES,0,buffer.length/9);
    return true;
  }
}
