class Datasource < ActiveRecord::Base
  acts_as_tree :order => "name"

  has_many :graph_datasources, :dependent => :destroy
  has_many :graphs, :through => :graph_datasources, :source => :graph, :order => "sortorder ASC", :dependent => :destroy
end
