varying vec4 v_CLR;
varying vec3 v_P;

uniform float display_normal;
uniform float radius_factor;
uniform float backface_cull;
uniform vec3 ambient;
uniform float exposure;

void main(void)
{
	//increment variable
	int i;

	// iterate through incoming points
	for(i=0; i< gl_VerticesIn; i++)
	{
		vec4 nr = gl_TexCoordIn[i][0];
		vec3 N = nr.xyz;
        float radius = nr.w * radius_factor;

		// pass through the encoded surface normal
		vec3 v_N = gl_NormalMatrix * nr.xyz;

        vec3 L = normalize(gl_LightSource[0].position.xyz); // get the camera
        float dotLN = dot(L, v_N);
        if ( backface_cull > 0 && dotLN < 0)
             continue;

        // let's do the lighting once
        v_CLR = gl_TexCoordIn[i][1] * ( vec4(ambient,0) + (1-ambient.x)*max(dot(L, v_N), 0.0) );

    	float pe = pow( 2, exposure );
	   	v_CLR *= vec4( pe,pe,pe, 1 );

	    vec3 dbg_nr = gl_TexCoordIn[i][1].xyz;

        gl_Position = gl_ModelViewProjectionMatrix * gl_PositionIn[i];
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * ( gl_PositionIn[i] + vec4( dbg_nr * radius, 0 ) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        EndPrimitive();
	}
}
