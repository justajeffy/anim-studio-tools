class Tab < ActiveRecord::Base
	acts_as_tree :order => "name"
  has_many :graphs, :dependent => :destroy
end
