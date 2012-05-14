require 'zlib'
require 'base64'
require 'time'

class ShowController < ApplicationController
  protect_from_forgery :only => [:update, :destroy]

  def index
    redirect_to :action => 'tab', :id => 2
  end

  # Load the main content based on selected tab 
  def tab

    # Simple ACL - put into conf
    admins = ['barry.robison', 'adminbr', 'kim.pearce', 'adminkp', 'jay.munzner', 'adminjm', 'kenny.ferreira', 'stephen.tanamal', 'adminst', 'admindw', 'david.ward', 'surendra.perera', 'adminsp']

    # Selected tab
    tabId = params[:id] ? params[:id] : 2

    # Check for user cookie and user's tab
    @username = cookies[:username] || ''
    checkUserTab(@username.gsub!("\"", ""))

    # Get tabs
	@navigation = navigation(tabId)

    # Get Graphs
    @graphs = graphs(tabId)

    # Get total graphs count
    @graphCount = Graph.find(:all).length
   
    # Check ACL
    @access = admins.include?(@username) ? true : false

    # Get total datasources used in graphs
    @dsCount = GraphDatasource.find(:all).length
  end

  # Show APC PDM reports
  def report

    # Config
	@pod = {
	        "Row B" => [
			             { "Rack 3 - Storage" => {
						                         "A6 L2" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_6b_module_6b.rrd"],
									             "B5 L2" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_5b_module_5b.rrd"]
						                       } },
	                     { "Chill" => {
						              "B1 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_1c_module_1c.rrd"],
						              "A4 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_4c_module_4c.rrd"]
						            } },
                         { "Rack 4 - Storage" => {
						                         "A6 L1" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_6a_module_6a.rrd"],
						                         "B5 L1" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_5a_module_5a.rrd"]
						                       } },
                         { "Chill" => {
                                      "B1 L1" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_1a_module_1a.rrd"],
                                      "A3 L2" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_3b_module_3b.rrd"]
						            } },
                         { "Rack 5 - Storage" => {
						                         "A6 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_6c_module_6c.rrd"],
						                         "B5 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_5c_module_5c.rrd"]
						                       } },
                         { "Chill" => {
						              "B1 L2" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_1b_module_1b.rrd"],
									  "A2 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_2c_module_2c.rrd"]
						            } },
                         { "Rack 6 - HP (Empty)" => {
						                            "A13" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_13a_module_13a.rrd"],
													"B7"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_7a_module_7a.rrd"],
													"A14" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_14a_module_14a.rrd"],
													"B8"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_8a_module_8a.rrd"]
						                          } },
                         { "Chill" => {
						              "B2 L2" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_2b_module_2b.rrd"],
									  "A3 L1" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_3a_module_3a.rrd"]
						            } },
                         { "Rack 7 - AL" => {
						                    "A16" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_16a_module_16a.rrd"],
											"B9"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_9a_module_9a.rrd"],
											"A15" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_15a_module_15a.rrd"],
											"B10" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_10a_module_10a.rrd"]
						                  } },
                         { "Chill" => {
						              "B2 L1" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_2a_module_2a.rrd"],
									  "A3 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_3c_module_3c.rrd"]
						            } },
                         { "Rack 8 - AL" => {
						                    "A17" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_17a_module_17a.rrd"],
											"B11" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_11a_module_11a.rrd"],
											"A18" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_18a_module_18a.rrd"],
											"B12" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_12a_module_12a.rrd"]
						                  } }

			           ],
	        "Row A" => [
			             { "Rack 3 - Storage/OTC" => {
						                         "B6 L1" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_6a_module_6a.rrd"],
									             "A5 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_5c_module_5c.rrd"]
						                       } },
	                     { "Chill" => {
						              "B2 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_2c_module_2c.rrd"],
						              "A1 L3" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_1c_module_1c.rrd"]
						            } },
                         { "Rack 4 - Storage" => {
						                         "A5 L1" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_5a_module_5a.rrd"],
						                         "B6 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_6c_module_6c.rrd"]
						                       } },
                         { "Chill" => {
                                      "B3 L1" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_3a_module_3a.rrd"],
                                      "A1 L1" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_1a_module_1a.rrd"]
						            } },
                         { "Rack 5 - Network" => {
						                         "A5 L2" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_5b_module_5b.rrd"],
						                         "B6 L2" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_6b_module_6b.rrd"]
						                       } },
                         { "Chill" => {
						              "B3 L2" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_3b_module_3b.rrd"],
									  "A1 L2" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_1b_module_1b.rrd"]
						            } },
                         { "Rack 6 - Dell" => {
						                            "A7" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_7a_module_7a.rrd"],
													"B13"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_13a_module_13a.rrd"],
													"A8" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_8a_module_8a.rrd"],
													"B14"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_14a_module_14a.rrd"]
						                          } },
                         { "Chill" => {
						              "B3 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_3c_module_3c.rrd"],
									  "A2 L2" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_2b_module_2b.rrd"]
						            } },
                         { "Rack 7 - HP Rack 1" => {
						                    "A9" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_9a_module_9a.rrd"],
											"B15"  => ["/opt/zenoss/perf/Devices/118.127.29.181/module_15a_module_15a.rrd"],
											"A10" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_10a_module_10a.rrd"],
											"B16" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_16a_module_16a.rrd"]
						                  } },
                         { "Chill" => {
						              "B4 L3" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_4c_module_4c.rrd"],
									  "A2 L1" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_2a_module_2a.rrd"]
						            } },
                         { "Rack 8 - HP (Empty)" => {
						                    "A12" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_12a_module_12a.rrd"],
											"B18" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_18a_module_18a.rrd"],
											"A11" => ["/opt/zenoss/perf/Devices/118.127.29.180/module_11a_module_11a.rrd"],
											"B17" => ["/opt/zenoss/perf/Devices/118.127.29.181/module_17a_module_17a.rrd"]
						                  } }

			           ]
	      }

    # Start and end times
    s = Time.parse(params[:start] || '').to_i
    e = Time.parse(params[:end] || '').to_i
    n = Time.parse('').to_i

    # Convert interval to seconds
    start_end_diff = e - s
    end_now_diff   = n - e

    # Price per kWh
    price = params[:price] || 0

    # Store the dates for later reference
    @times = []

    # Keep totals per row and overall
    @totals = [{}, 0, {}]

    @pod.each { |row, r|
		@pod[row].each { |rack|
			rack.each { |label, circuits|
				circuits.each { |circuit, rrd|
				
					data = {}
					outp = IO.popen("/usr/bin/rrdtool fetch #{rrd[0]} AVERAGE -s -#{start_end_diff + end_now_diff}s -e -#{end_now_diff}")

					# Clean up the data
					outp.readlines.each { |line|

						# Each line has to start with the epoch
						next if !line.match(/^\d/)

						# Chomp chomp
						line.chomp

						# Split for date and value
						epoch, value = line.split(": ")

						time  = Time.at(epoch.to_i).strftime("%d.%m.%Y")
						value = value.to_f * 300 * 0.1

						# Add the date if it is not yet in the lookup list
						@times.push(time) if not @times.include?(time)

						# Sum up the values per day
						data[time] = data[time] ? data[time] + value : value

						# Sum up the values per day for all modules
						@totals[0][row] = @totals[0][row] ? @totals[0][row] + value : value

						# Total total
						@totals[1] += value

                        # Total price
                        @totals[2][row] = @totals[2][row] ? @totals[2][row] + value.to_f * price.to_f : value.to_f * price.to_f
					}

					rrd.push(data)
				}
			}
		}
	}

  end

  def newGraph

    # Check if this is an actual save or just opening the form
    if not params[:save]
      @id   = params[:id]
      @type = 'new'

      render :partial => 'newGraph'
    else
     
      # Get the selected tab
      tab = Tab.find(params[:id])

      graph = Graph.create(:title => params[:title],
                           :vlabel => params[:vlabel],
                           :width => (params[:width] == '') ? 1000 : params[:width],
                           :height => (params[:height] == '') ? 140 : params[:height],
                           :period => periodToSecs(params[:period]),
                           :upperlimit => params[:upperlimit],
                           :lowerlimit => params[:lowerlimit],
                           :stacked => (params[:stacked] == 'on') ? true : false,
                           :legend => (params[:legend] == 'on') ? true : false,
                           :max => (params[:graphmax] == 'on') ? true : false,
                           :tab_id => tab.id)
      
      params[:datasources].each do |datasource|
        GraphDatasource.create(:graph_id => graph.id,
                               :datasource_id => datasource)
      end

      # Redirect to selected tab
      redirect_to :action => 'tab', :id => params[:id]
    end                                           
  end

  def editGraph

    @graph = Graph.find(params[:id])

    if not params[:save]
      @type = 'edit'

      render :partial => 'editGraph'
    else
      
      @graph.attributes = { :title => params[:title],
                            :vlabel => params[:vlabel],
                            :width => params[:width],
                            :height => params[:height],
                            :period => periodToSecs(params[:period]),
                            :upperlimit => params[:upperlimit],
                            :lowerlimit => params[:lowerlimit],
                            :stacked => (params[:stacked] == 'on') ? true : false,
                            :legend => (params[:legend] == 'on') ? true : false,
                            :max => (params[:graphmax] == 'on') ? true : false }

      # Store newly added datasources
      if params[:datasources]
        params[:datasources].each do |datasource|
          GraphDatasource.create(:graph_id => @graph.id,
                                 :datasource_id => datasource,
                                 :negative => 0)
        end
      end

      # Change the sign if necessary
      if params[:datasources_signs]
        params[:datasources_signs].each do |dss|
          ds, sign = dss.split("_")

          sign = (sign == 'negative') ? 1 : 0

          # Find the GraphDatasource
          graphDatasource = GraphDatasource.find_by_graph_id_and_datasource_id(@graph.id, ds)
          graphDatasource.negative = sign
          graphDatasource.save
        end
      end

      # Redirect to selected Tab
      if @graph.save
        redirect_to :action => 'tab', :id => @graph.tab.id
      end
    end
  end

  def dsAvailable
    
    # Get the children of the selected node, or of the root node
    @datasources = Datasource.find(params[:id]||1).children

    render :partial => 'dsAvailable'
  end

  def dsUsed
    
    id = (params[:type] == 'edit') ? params[:id] : nil

    # Get the selected graph
    if id
      @graph = Graph.find(id)
    end

    render :partial => 'dsUsed'
  end

  def sortDatasources

    graphId     = params[:graphid]
    datasources = params[:data].split("&")
    counter     = 0

    # Parse the datasources list ...
    datasources.each do |datasource|
      ds  = datasource.split("=")[1]
      gds = GraphDatasource.find_by_graph_id_and_datasource_id(graphId, ds)

      gds.sortorder = counter
      gds.save

      counter += 1
    end

    # Nothing to render
    render :nothing => true
  end

  def sortGraphs
    
    graphs  = params[:data].split("&")
    counter = 0
    
    # For each graph id find the graph and set new sortorder
    graphs.each do |graph|
      graph = Graph.find(graph.split("=")[1])
      
      graph.sortorder = counter
      graph.save
      
      counter += 1
    end
    
    # Nothing to render
    render :nothing => true
  end

  def copyGraph
    
    graphId = params[:id]
    tabId   = params[:tabId]
    
    # Get the graph to be copied
    cGraph = Graph.find(graphId)
    
    graph = Graph.create(:title => cGraph.title,
                         :vlabel => cGraph.vlabel,
                         :width => cGraph.width,
                         :height => cGraph.height,
                         :period => cGraph.period,
                         :upperlimit => cGraph.upperlimit,
                         :lowerlimit => cGraph.lowerlimit,
                         :stacked => cGraph.stacked,
                         :legend => cGraph.legend,
                         :max => cGraph.max,
                         :sortorder => cGraph.sortorder,
                         :tab_id => tabId)
                         
    # Create copies of the GraphDatsources
    cGraph.graph_datasources.each do |graphDatasource|
      GraphDatasource.create(:graph_id => graph.id,
                             :datasource_id => graphDatasource.datasource_id,
                             :negative => graphDatasource.negative,
                             :sortorder => graphDatasource.sortorder)
    end
    
    # Redirect to tab containing the copied graph
    redirect_to :action => 'tab', :id => tabId
  end


  def removeDs
    GraphDatasource.find_by_graph_id_and_datasource_id(params[:gr], params[:ds]).destroy

    render :nothing => true
  end

  def newTab
    if not params[:save]
      @id    = params[:id]
      @level = params[:level]

      render :partial => 'newTab'
    else
      
      # Find the parent if on same level otherwise create tab below selected tab
      parent = (params[:level] == 'same') ? Tab.find(params[:id]).parent : Tab.find(params[:id])
            
      tab = Tab.create(:name => params['tabName'],
                       :parent_id => parent.id)
            
      # Redirect to newly created tab
      redirect_to :action => 'tab', :id => tab.id
    end
  end
  
  def deleteTab
    if not params[:delete]
      @id = params[:id]
      
      render :partial => 'deleteTab'
    else
      tab = Tab.find(params[:id])
      
      # Get the parent to load once tab is deleted
      parent = tab.parent
      
      # Delete tab and anything below it
      tab.destroy
      
      redirect_to :action => 'tab', :id => parent
    end
  end

  def deleteGraph
    if not params[:delete]
      @id = params[:id]
      
      render :partial => 'deleteGraph'
    else
      graph = Graph.find(params[:id])
      tab   = graph.tab
      
      # Delete graph and datasources
      graph.destroy
      
      redirect_to :action => 'tab', :id => tab
    end  
  end

  def renameTab
    if not params[:save]
      @id = params[:id]
      @tabname = Tab.find(@id).name
      
      render :partial => 'renameTab'
    else
      
      # Change the name of the tab
      tab = Tab.find(params[:id])
      tab.name = params[:tabName]
      tab.save
      
      redirect_to :action => 'tab', :id => tab
    end
  end

  private
  def navigation(selected)

    # Arrays to keep level details
    levels = []

    # The currently selected tab
    selected = Tab.find(selected)

    # Get ancestors of currently selected tab
    ancestors = selected.ancestors.reverse

    # Get siblings of each ancestor and put into levels array
    ancestors.each { |ancestor| levels.push(ancestor.self_and_siblings) if not ancestor.self_and_siblings.empty? }

    # Put siblings of currently selected tab into levels array
    levels.push(selected.self_and_siblings)

    # Put possible children of currently selected tab in levels array
    levels.push(selected.children) if not selected.children.empty?

    # Add currently selected tab to ancestors array for later lookup
    ancestors.push(selected)
    
    return {"levels" => levels, "active" => ancestors}
  end

  def graphs(id)

    # The Zenoss RenderServer location
    renderserver = "http://zenoss02.drd.int:8080/zport/RenderServer/render?gopts="

    # List of graph to be returned
    graphlist = []

    # User defined values from width|height|duration|end fields
    userWidth    = params[:width] == '' ? nil : params[:width]
    userHeight   = params[:height] == '' ? nil : params[:height]
    userDuration = params[:duration] == '' ? nil : params[:duration]
    userEnd      = params[:end] == '' ? nil : params[:end]

    # Set up colorscheme - should really be set somewhere else
    areas = ['#500000', '#877700', '#005000', '#0000D0', '#666666', '#996600', 
             '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00',
             '#500000', '#877700', '#005000', '#0000D0', '#666666', '#996600', 
             '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00', 
             '#500000', '#877700', '#005000', '#0000D0', '#666666', '#996600', 
             '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00', 
             '#500000', '#877700', '#005000', '#0000D0', '#666666', '#996600', 
             '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00']

    colors = ['#E00000', '#E0E000', '#00E000', '#0000D0', '#666666', '#996600', 
              '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ffee',
              '#9395CC', '#0E9005', '#FFAA00', '#929000',
              '#E00000', '#E0E000', '#00E000', '#0000D0', '#666666', '#996600', 
              '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00', 
              '#E00000', '#E0E000', '#00E000', '#0000D0', '#666666', '#996600', 
              '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00', 
              '#E00000', '#E0E000', '#00E000', '#0000D0', '#666666', '#996600', 
              '#66CCFF', '#339933', '#3366cc', '#ffcc99', '#33AACC', '#00ff00']

    #colorscheme = '|-cBACK#000000|-cCANVAS#170047|-cSHADEA#505050|-cSHADEB#D0D0D0|-cGRID#505050|-cMGRID#909090|-cFONT#D0D0D0'
    colorscheme = '|-cBACK#000000|-cCANVAS#000000|-cSHADEA#FFFFFF|-cSHADEB#FFFFFF|-cGRID#565656|-cMGRID#354e64|-cFONT#FFFFFF'

    # Find all graphs for currently selected tab
    graphs = Tab.find(id).graphs.find(:all, :order => 'sortorder ASC')

    # Generate the necessary data for each graph
    graphs.each do |graph|
    
      # Check to use user defined of database values
      width    = userWidth    ? userWidth  : graph.width
      height   = userHeight   ? userHeight : graph.height
      duration = userDuration ? periodToSecs(userDuration) : graph.period
      graphEnd = userEnd      ? periodToSecs(userEnd) : 0

      # General graph settings
      lineType = graph.stacked ? 'AREA'   : 'LINE';
      stack    = graph.stacked ? ':STACK' : '';
      rigid    = (graph.upperlimit && graph.lowerlimit) ? '|--rigid' : ''
      counter  = 0

      # Re-sort the datasources in case of positive/negative stacking
      positives = []
      negatives = []
      graph.graph_datasources.find(:all, :order => "sortorder ASC").each do |graphDatasource|
        graphDatasource.negative ? negatives.push(graphDatasource) : positives.push(graphDatasource)
      end

      # Join the two arrays
      graphDatasources = positives | negatives

      # Start the graph options string with general settings
      gopts  = "-F|-E|--height=#{height}#{colorscheme}|--upper-limit=#{graph.upperlimit}"
      gopts += "|--lower-limit=#{graph.lowerlimit}#{rigid}|--vertical-label=#{graph.vlabel}|--title=#{graph.title}"
      gopts += "|--font=DEFAULT:8:Liberation Mono"

      # Add all datasources to graph
      graphDatasources.each do |graphDatasource|

        datasource = graphDatasource.datasource

        # Set the legendname for this datasource
        legendname = datasource.name

        # Custom calculations
        nagative, loadAvg, memory, rawCpu, rawCpuEx, netAppBytes, blueArcBytes, osxBytes, pdm, pdm_total_power = ''
        negative = ",-1,*"   if graphDatasource.negative
        loadAvg  = ",100,/"  if legendname.match(/laLoadInt/)
        memory   = ",1024,*" if legendname.match(/_mem/)
        rawCpu   = ",10,/"   if legendname.match(/_ssCpuRaw/)
        rawCpuEx = ",10,*"   if (legendname.match(/^c0/) && legendname.match(/_ssCpuRaw/) || legendname.match(/ganglia01_ssCpuRaw/) || legendname.match(/itwiki01_ssCpu/) || legendname.match(/farm01_ssCpu/) || legendname.match(/zenoss01_ssCpuRaw/) || legendname.match(/builder01_ssCpuRaw/) || legendname.match(/im01_ssCpuRaw/) || legendname.match(/puppet02_ssCpuRaw/) || legendname.match(/ams01_ssCpuRaw/) || legendname.match(/farmtest01_ssCpuRaw/))

        farmdb01   = ",10,*,4,/"  if legendname.match(/farmdb01_ssCpuRaw/)
        sql02      = ",10,*,4,/"  if legendname.match(/sql02_ssCpuRaw/)
        zenoss02   = ",10,*,2,/"  if legendname.match(/zenoss02_ssCpuRaw/)
        twocore    = ",10,*,2,/"  if legendname.match(/shotgundb01_ssCpuRaw/) || legendname.match(/sql01_ssCpuRaw/)
        fourcore   = ",10,*,4,/"  if legendname.match(/shotgun03_ssCpuRaw/)
        sixcore    = ",10,*,6,/"  if legendname.match(/z600s.*_ssCpuRaw/)
        eightcore  = ",10,*,8,/"  if legendname.match(/om.*_ssCpuRaw/)
        twelvecore = ",10,*,12,/" if legendname.match(/z600d.*_ssCpuRaw/)

        # Calculate bytes for Bluearc storage ...
        blueArcBytes = ",1024,/,1024,/,1000000000,*" if legendname.match(/^bluearc/) && ( legendname.match(/Blocks$/) || legendname.match(/snapshots$/) )

        # Rename datasource for Blue Arc storage 
        legendname.gsub!(legendname.split("_").last, 'total') if legendname.match(/^bluearc/) && legendname.match(/totalBlocks$/)
        legendname.gsub!(legendname.split("_").last, 'avail') if legendname.match(/^bluearc/) && legendname.match(/availBlocks$/)
        legendname.gsub!(legendname.split("_").last, 'used')  if legendname.match(/^bluearc/) && legendname.match(/usedBlocks$/)

        # Calculate bytes for netapp storage ...
        netAppBytes = ",1024,/,1024,/,1000000000,*" if legendname.match(/^filer/) && legendname.match(/Blocks$/)

        # Rename datasource for netApp storage and compensate for Zenoss naming bug
        legendname.gsub!(legendname.split("_").last, 'total') if legendname.match(/^filer/) && legendname.match(/totalBlocks$/)
        legendname.gsub!(legendname.split("_").last, 'used') if legendname.match(/^filer/) && legendname.match(/usedBlocks$/)
        legendname.gsub!(legendname.split("_").last, 'avail')  if legendname.match(/^filer/) && legendname.match(/availBlocks$/)

        # Calculate bytes for OS X Volumes ...
        osxBytes = ",8192,*,1024,/,1024,/,1024,/,1000000000,*" if legendname.match(/^finalcutserver/) && legendname.match(/usedBlocks$/)

        # Adjust the PDM counters to not average per second
        pdm = ",300,*,0.1,*" if legendname.match(/^PDM-A_module/) || legendname.match(/^PDM-B_module/)

        # Adjust the PDM total power gauges 
		pdm_total_power = ",10,*" if legendname.match(/^PDM-A_total_power/) || legendname.match(/^PDM-B_total_power/)

        # Custom renames of datasources for better readability
        legendname.gsub!(legendname.split("_").last, 'Inbound')  if legendname.match(/InOctets/)
        legendname.gsub!(legendname.split("_").last, 'Outbound') if legendname.match(/OutOctets/)

        # Get the ds name from the rrd file
        ds = `/var/www/graphite/script/dsName.sh #{datasource.rrd}`.strip()

        gopts += "|DEF:#{datasource.name}-raw=#{datasource.rrd}:#{ds}:AVERAGE|"
        gopts += "CDEF:#{datasource.name}-custom=#{datasource.name}-raw#{negative}#{loadAvg}#{memory}#{rawCpu}#{rawCpuEx}#{netAppBytes}#{blueArcBytes}#{osxBytes}#{pdm}#{pdm_total_power}#{farmdb01}#{sql02}#{zenoss02}#{twocore}#{fourcore}#{sixcore}#{eightcore}#{twelvecore}|"
        gopts += "CDEF:#{datasource.name}-legendvalue=#{datasource.name}-raw#{loadAvg}#{memory}#{rawCpu}#{rawCpuEx}#{netAppBytes}#{blueArcBytes}#{osxBytes}#{pdm}#{pdm_total_power}#{farmdb01}#{sql02}#{zenoss02}#{twocore}#{fourcore}#{sixcore}#{eightcore}#{twelvecore}|"
        gopts += "CDEF:#{datasource.name}=#{datasource.name}-custom|"

        # Check if graph max was set and act accordingly
        if graph.max
          gopts += "DEF:#{datasource.name}-area-raw=#{datasource.rrd}:#{ds}:MAX|"
          gopts += "CDEF:#{datasource.name}-area-custom=#{datasource.name}-area-raw#{negative}#{loadAvg}#{memory}#{rawCpu}#{rawCpuEx}#{netAppBytes}#{blueArcBytes}#{osxBytes}#{pdm}#{pdm_total_power}#{farmdb01}#{sql02}#{zenoss02}#{twocore}#{fourcore}#{sixcore}#{eightcore}#{twelvecore}|"
          gopts += "CDEF:#{datasource.name}-area=#{datasource.name}-area-custom|"
          gopts += "AREA:#{datasource.name}-area" + areas[counter] + "|"

          if graph.legend
            gopts += "LINE:#{datasource.name}" + colors[counter] + ":" + legendname.ljust(65) + "|";
          else
            gopts += "LINE:#{datasource.name}" + colors[counter] + "|";
          end
        else
          # Check for negative stacking - if a DS is not set to stack but following DS are then it starts a new stack
          # One stack for positives and one stack for negatives
          stacking = ((positives[0] && graphDatasource.id == positives[0].id) || (negatives[0] && graphDatasource.id == negatives[0].id)) ? '' : stack

          if graph.legend
            gopts += "#{lineType}:#{datasource.name}" + colors[counter] + ":" + legendname.ljust(65) + "#{stacking}|"
          else
            gopts += "#{lineType}:#{datasource.name}" + colors[counter] + ":#{stacking}|"
          end
        end

        if graph.legend
          gopts += "GPRINT:#{datasource.name}-legendvalue:LAST: cur\\: %6.2lf%s|"
          gopts += "GPRINT:#{datasource.name}-legendvalue:MIN: min\\: %6.2lf%s|"
          gopts += "GPRINT:#{datasource.name}-legendvalue:AVERAGE: avg\\: %6.2lf%s|"
          gopts += "GPRINT:#{datasource.name}-legendvalue:MAX: max\\: %6.2lf%s\\j"
        else
          gopts += "COMMENT:\s"
        end

        # Increase the counter for next color
        counter += 1
      end

      # The actual graph is generated by the Zenoss RenderServer.
      # This expects an encoded/compressed string and returns an image.
      # For each graph generate this encoded string.
      deflate = Zlib::Deflate.new(9).deflate(gopts, Zlib::FINISH)
      encode  = Base64.b64encode(deflate)
      eopts   = encode.gsub('+', '-').gsub('/', '_').gsub("\n", '')

      # Format start and end time
      startTime = (Time.now - (graphEnd.to_i + duration.to_i).seconds).to_s
      startTime = startTime.gsub(" ", "%20").gsub(":", "%5C%3A")

      endTime = (Time.now - graphEnd.to_i.seconds).to_s
      endTime = endTime.gsub(" ", "%20").gsub(":", "%5C%3A")

      additional = "&drange=#{duration}&width=#{width}&start=end-#{duration}&end=now-#{graphEnd}s&comment=#{startTime}%20to%20#{endTime}"

      img = "#{renderserver}#{eopts}#{additional}"

      graphlist.push([graph.id, graph.title, img])
    end

    return graphlist || []
  end

  def periodToSecs(period)
    
    matches = period.match(/[0-9]+/)

    if matches
      result = period.split(matches[0])
      factor = 0

      if not result.empty?
        factor = 1        if result[1].match(/^s/) # seconds
        factor = 3600     if result[1].match(/^h/) # 60*60 - hour
        factor = 86400    if result[1].match(/^d/) # 60*60*24 - day
        factor = 604800   if result[1].match(/^w/) # 60*60*24*7 - week
        factor = 2592000  if result[1].match(/^m/) # 60*60*24*30 - month
        factor = 31536000 if result[1].match(/^y/) # 60*60*24*365 - year
      end

      return matches[0].to_i * factor
    end

    return 86400 # 24 hours
  end
 
  def checkUserTab(username)
    
    # Check the DB if this user tab already exists
    if not tab = Tab.find_by_name_and_parent_id("user_#{username}", 1)
      
      # Store the username in DB
      tab = Tab.create(:name => "user_#{username}",
                       :parent_id => 1)
    end
  end  
end
