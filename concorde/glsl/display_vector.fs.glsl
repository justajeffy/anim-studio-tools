varying vec4 v_CLR;
varying vec3 v_P;

uniform sampler3D TextureSampler0;
uniform vec4 wipe_values;

void main ()
{
    if ( !((v_P.x >= wipe_values.x) && (v_P.x <= wipe_values.y)) ) discard;
    if ( !((v_P.y >= wipe_values.z) && (v_P.y <= wipe_values.w)) ) discard;

    // lighting done in the geometry shader
    gl_FragColor = vec4(v_CLR.xyz, 1);
    gl_FragColor = OCIODisplay( gl_FragColor, TextureSampler0 );
}
