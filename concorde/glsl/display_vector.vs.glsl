uniform float display_normal;

void main ()
{
	// don't transform here, we'll do it in geo shader
	gl_Position = gl_Vertex;

	// pass through the texture coord
	gl_TexCoord[0] = gl_MultiTexCoord0;

    if (display_normal != 0)   gl_TexCoord[1] = vec4( gl_MultiTexCoord0.xyz, gl_MultiTexCoord1.w );
    else                       gl_TexCoord[1] = gl_MultiTexCoord1;
}
