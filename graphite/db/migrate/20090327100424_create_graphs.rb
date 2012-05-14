class CreateGraphs < ActiveRecord::Migration
  def self.up
    create_table :graphs do |t|
			t.column :title, :string
			t.column :vlabel, :string
			t.column :width, :integer
			t.column :height, :integer
			t.column :period, :integer
			t.column :upperlimit, :decimal
			t.column :lowerlimit, :decimal
			t.column :stacked, :boolean
			t.column :max, :boolean
			t.column :sortorder, :integer
			t.column :tab_id, :integer
    end
  end

  def self.down
    drop_table :graphs
  end
end
