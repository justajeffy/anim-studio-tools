class GraphDatasource < ActiveRecord::Base
  belongs_to :graph
  belongs_to :datasource
end
