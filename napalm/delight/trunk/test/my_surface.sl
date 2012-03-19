surface my_surface(	color col = color(1,1,1);
					string texturename = "";
					uniform string my_strings[] = {};
					output varying float my_radius = 0.75)
{
	float prim_type = -1;
	attribute( "user:prim_type", prim_type );

	uniform float i = 0;
	for( i = 0; i < arraylength( my_strings ); i+=1 ){
		printf( "INFO: my_strings[%.0f]: %s\n", i, my_strings[i] );
	}

	Oi = Os;

	if( prim_type > 0.5 ){
		Ci = col * Cs * texture( texturename, s, t ) * Oi;
		my_radius = 4.0 * (1.0 - float texture( texturename, s, t ) * 0.9);

		Ci = color(transform( "world", P )) * Oi;
		//Ci = color(0,1,1);
	} else {
		//Ci = color(0,0,1);
		Ci = color(transform( "world", P )) * Oi;
	}

	// evaluating inside sx context
	if( prim_type < -0.5 ){
		// Ci = color(transform( "world", P )) * Oi;
		Ci = color(P) * Oi;
	}

}
