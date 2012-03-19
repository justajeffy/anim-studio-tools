varying vec3 v_N;

void main () {
	gl_Position = ftransform();
	v_N = gl_NormalMatrix * gl_Normal;
	gl_TexCoord[0] = gl_MultiTexCoord0;
}