varying vec4 v_N;
varying vec4 v_USRVAR;
varying vec3 v_P;

uniform vec3 ambient;
uniform float exposure;
uniform float backface_cull;
uniform float disc_point_distance_transition;
uniform vec4 wipe_values;

uniform sampler3D TextureSampler0;

void main ()
{
    if (v_N.w < disc_point_distance_transition) discard;

    if ( !((v_P.x >= wipe_values.x) && (v_P.x <= wipe_values.y)) ) discard;
    if ( !((v_P.y >= wipe_values.z) && (v_P.y <= wipe_values.w)) ) discard;

	vec3 N = normalize(v_N.xyz);

	vec3 L = normalize(gl_LightSource[0].position.xyz);
	vec3 diffuse = (1.0 - ambient) * max(dot(L, N), 0.0) * gl_LightSource[0].diffuse.xyz;

	if ( backface_cull > 0 && dot(L, N) < 0)
	     discard;

	gl_FragColor = vec4(ambient + diffuse, 1.0);
	gl_FragColor *= v_USRVAR;

	float pe = pow( 2, exposure );
	gl_FragColor *= vec4( pe,pe,pe, 1 );
	gl_FragColor = OCIODisplay( gl_FragColor, TextureSampler0 );
}
