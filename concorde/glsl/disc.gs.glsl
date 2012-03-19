varying vec4 v_CLR;
varying vec3 v_P;

uniform float radius_factor;
uniform vec3 ambient;
uniform float exposure;
uniform float disc_point_distance_transition;
uniform float backface_cull;

void main(void)
{
	//increment variable
	int i;

    vec2 P0 = vec2( -0.0, 1.0 );
    vec2 P1 = vec2( -0.866025404019, 0.499999999593 );
    vec2 P2 = vec2( -0.866025403314, -0.500000000814 );
    vec2 P3 = vec2( 1.41020693977e-09, -1.0 );
    vec2 P4 = vec2( 0.866025404725, -0.499999998372 );
    vec2 P5 = vec2( 0.866025402609, 0.500000002035 );

	// iterate through incoming points
	for(i=0; i< gl_VerticesIn; i++)
	{
        vec4 centerMV = gl_ModelViewMatrix * gl_PositionIn[i];
        if (centerMV.z > 0) continue; // skip what is behind the camera

        float dist2Cam = -centerMV.z;
        if (dist2Cam > disc_point_distance_transition) continue;

		vec4 nr = gl_TexCoordIn[i][0];
		vec3 N = nr.xyz;
		float radius = nr.w * radius_factor;

		// pick an arbitrary up vector
		vec3 UP = vec3(0,1,0);
		if( abs( dot( N, UP ) ) > 0.9 ){
			UP = vec3(1,0,1);
		}

		// the axes of our disc
		vec3 across_a = normalize( cross( N, UP ) ) * radius;
		vec3 across_b = normalize( cross( N, across_a ) ) * radius;

		// transform the normal
		vec3 v_N = gl_NormalMatrix * nr.xyz;

        vec3 L = normalize(gl_LightSource[0].position.xyz);

        if ( backface_cull > 0 && dot(L, v_N) < 0)
            continue;

        // let's do the lighting once
        v_CLR = gl_TexCoordIn[i][1] * ( vec4(ambient,0) + (1-ambient.x)*max(dot(L, v_N), 0.0) );

    	float pe = pow( 2, exposure );
    	v_CLR *= vec4( pe,pe,pe, 1 );

        vec4 centerMVP = gl_ModelViewProjectionMatrix * gl_PositionIn[i];

		// emit a disc of triangles

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P5.x * vec4(across_a,0) + P5.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P0.x * vec4(across_a,0) + P0.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P4.x * vec4(across_a,0) + P4.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P2.x * vec4(across_a,0) + P2.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P3.x * vec4(across_a,0) + P3.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        // end primitive ( triangle strip )
        EndPrimitive();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P0.x * vec4(across_a,0) + P0.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P1.x * vec4(across_a,0) + P1.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        gl_Position = gl_ModelViewProjectionMatrix * (gl_PositionIn[i] + P2.x * vec4(across_a,0) + P2.y * vec4(across_b,0) );
        v_P.xyz = gl_Position.xyz / gl_Position.w;
        EmitVertex();

        // end primitive ( triangle strip )
        EndPrimitive();
	}

}
