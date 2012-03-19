varying vec4 v_USRVAR;
uniform float channel_size;
uniform float display_position;
uniform float display_normal;
uniform float display_radius;

void main () {
	// don't transform here, we'll do it in geo shader
	gl_Position = gl_Vertex;

	// pass through the texture coord
	gl_TexCoord[0] = gl_MultiTexCoord0;

    if (display_position != 0)      gl_TexCoord[1] = gl_Vertex;
    else if (display_normal != 0)   gl_TexCoord[1] = gl_MultiTexCoord0;
    else if (display_radius != 0)   gl_TexCoord[1] = gl_MultiTexCoord0.wwww;
    else if (channel_size == 0)     gl_TexCoord[1] = vec4(1,1,1,1);
    else if (channel_size == 1)     gl_TexCoord[1] = gl_MultiTexCoord1.xxxx;
    else if (channel_size == 2)     gl_TexCoord[1] = vec4( gl_MultiTexCoord1.xy, 0, 0 );
    else                            gl_TexCoord[1] = gl_MultiTexCoord1;
}
