uniform sampler2D TextureSampler0;

varying vec3 v_N;

void main () {
	float amb = 0.05;
	//vec3 color = vec3(1,0.9,0.8);

	vec2 oTexCoord = vec2( gl_TexCoord[0].x, gl_TexCoord[0].y );
	vec3 color = texture2D( TextureSampler0, oTexCoord );

	vec3 N = normalize(v_N);
	vec3 L = normalize(gl_LightSource[0].position.xyz);
	vec3 H = normalize(gl_LightSource[0].halfVector.xyz);

	vec3 ambient = color * amb;
	vec3 diffuse = color * (1.0 - amb) * max(dot(L, N), 0.0) * gl_LightSource[0].diffuse;
	diffuse *= 0.7;
	//vec3 specular = vec3(1.0, 1.0, 1.0) * pow(max(dot(H, N), 0.0), 16.0);
	vec3 specular = vec3(0,0,0);

	gl_FragColor = vec4(ambient + diffuse + specular, 1.0);
}
