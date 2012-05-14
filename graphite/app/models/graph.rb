class Graph < ActiveRecord::Base
  belongs_to :tab

  has_many :graph_datasources, :dependent => :destroy
  has_many :datasources, :through => :graph_datasources, :order => "sortorder ASC", :dependent => :destroy
end
