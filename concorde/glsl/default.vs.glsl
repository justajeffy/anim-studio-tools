varying vec3 v_P;

void main ()
{
    gl_Position = ftransform();
    v_P.xyz = gl_Position.xyz / gl_Position.w;
}
