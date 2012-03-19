varying vec4 v_N;
varying vec4 v_USRVAR;
varying vec3 v_P;

uniform float channel_size;
uniform float display_position;
uniform float display_normal;
uniform float display_radius;
uniform float radius_factor;

void main ()
{
    vec4 centerMV = gl_ModelViewMatrix * gl_Vertex;
    v_N.w = -centerMV.z;

    if (display_position != 0)      v_USRVAR = gl_Vertex;
    else if (display_normal != 0)   v_USRVAR = gl_MultiTexCoord0;
    else if (display_radius != 0)   v_USRVAR = radius_factor * gl_MultiTexCoord0.wwww;
    else if (channel_size == 0)     v_USRVAR = vec4(1,1,1,1);
    else if (channel_size == 1)     v_USRVAR = gl_MultiTexCoord1.xxxx;
    else if (channel_size == 2)     v_USRVAR = vec4( gl_MultiTexCoord1.xy, 0, 0 );
    else                            v_USRVAR = gl_MultiTexCoord1;

    gl_Position = ftransform();
    v_N.xyz = gl_NormalMatrix * gl_MultiTexCoord0.xyz;
    v_P.xyz = gl_Position.xyz / gl_Position.w;
}
