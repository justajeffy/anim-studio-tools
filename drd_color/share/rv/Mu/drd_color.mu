
module: drd_color
{

//
use rvtypes;
use commands;
use app_utils;
use extra_commands;
use io;
use string;
//
require system;

class: CustomColorManagementMode : MinorMode
{

    string m_DRD_DISPLAYLUT_RV;
    string m_DRD_COLOR_ROOT;

    //For some reason mu is failing where user dont have access to one folder and mu is trying check a file inside that
    //folder in that case mu will simply fail . so just added a another function to catch that error.

	method: checkFileSystem(bool; string file_path)
	{
		bool file_stat = false;
		try
		{
			if (path.exists(file_path))
			{
				file_stat = true;
			}
		}
		catch(...)
		{
			file_stat = false;
		}
		return file_stat;
	}

    //
    //  This function is bound to the new-source event. Its called any
    //  time a new source is added to RV. You can set up prefered color space
    //  conversions and pixel aspect, etc here. If the file name is meaningful
    //  you can even use that.


    method: sourceSetup (void; Event event)
    {

        let args      = event.contents().split(";;"),
            source    = args[0],
            type      = args[1],
            file      = args[2],
            exrRE     = regex(".*\\.(((s|e)xr)|((S|E)XR))$"),
            dpxRE 	  = regex(".*\\.(((s|d)px)|((S|D)PX))$");

      	m_DRD_DISPLAYLUT_RV = system.getenv("DRD_DISPLAYLUT_RV");
      	//for some reason this var not there in arsenal env list so hard coding root path
      	m_DRD_COLOR_ROOT = system.getenv("DRD_DRD_COLOR_ROOT");
      	//moving home folder up then it can be accessed inside the try
      	string homedir=system.getenv("HOME");

        // Apply DRD_DISPLAYLUT_RV on EXR files only

        bool isEXR=exrRE.match(file);
        bool isDPX=dpxRE.match(file);

        if (isEXR || isDPX)
        {

            try {

            	//Calibration config file
            	string calib_config=homedir+"/.config/calib.yaml";

            	if (!checkFileSystem(calib_config))
            	{
            		print( "Calib config file '%s' not found. Assuming sRGB.\n" % calib_config);
					if (isEXR)
					{
        				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383CoolGrade_ocio.csp";
        			}
        			else
        			{
        				if (isDPX)
        				{
            				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383_DPX_ocio.csp";
            			}
        			}
            	}
            	else
            	{
	            	let file	   = ifstream(calib_config),
    	        		everything = read_all(file),
        	    		lines	   = everything.split("\n\r");

					if ( lines[1] == "target: DCI" )
					{
						if (isEXR)
						{
            				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383CoolGrade_ocio_Dreamcolor.csp";
            			}
            			else
            			{
            				if (isDPX)
            				{
	            				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383_DPX_ocio_Dreamcolor.csp";
	            			}
            			}
            		}
            		else
            		{
						if (isEXR)
						{
            				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383CoolGrade_ocio.csp";
            			}
            			else
            			{
            				if (isDPX)
            				{
	            				m_DRD_DISPLAYLUT_RV = m_DRD_COLOR_ROOT + "/share/ocio/hf2/files/Kodak2383_DPX_ocio.csp";
	            			}
            			}
            		}
            		file.close();
            	}

                print( "DRD INFO: DRD_DISPLAYLUT_RV = %s\n" % m_DRD_DISPLAYLUT_RV );

                try {
                    readLUT(m_DRD_DISPLAYLUT_RV, "#RVDisplayColor");
                    setIntProperty("#RVDisplayColor.lut.active", int[] {1});
                    updateLUT();
                    setIntProperty("#RVDisplayColor.color.sRGB", int[] {0});
                }
                catch(...)
                {
                    displayFeedback("LUT: %s Failed To Load" % m_DRD_DISPLAYLUT_RV, 3.0);
                    print("ERROR: LUT: %s Failed To Load. File may not exist\n" % m_DRD_DISPLAYLUT_RV);
                }
                redraw();
            }
            catch(...)
            {
                // fall back to inbuild color function if the lut env is not set
                displayFeedback("DRD_DISPLAYLUT_RV is not set in the env", 3.0);
                print("ERROR: DRD_DISPLAYLUT_RV is not set in the env\nOr you dont have access to %s\n"%(homedir));
            }

        }

        //
        // This is done to allow other functions bound to this event
        // to get a chance to modify the state as well.
        event.reject();

    }

    method: CustomColorManagementMode (CustomColorManagementMode;)
    {
        init("Custom Color Management",
             nil,
             [("new-source", sourceSetup, "DRD Color Setup")],
             nil);
    }
}

\: createMode (Mode;)
{
    return CustomColorManagementMode();
}

} // end module

