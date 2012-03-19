varying vec3 v_P;
uniform vec3 ambient;
uniform vec4 wipe_values;
void main ()
{
    if ( !((v_P.x >= wipe_values.x) && (v_P.x <= wipe_values.y)) ) discard;
    if ( !((v_P.y >= wipe_values.z) && (v_P.y <= wipe_values.w)) ) discard;

	gl_FragColor = vec4(ambient, 1.0);
}
