class CreateGraphDatasources < ActiveRecord::Migration
  def self.up
    create_table :graph_datasources do |t|
      t.column :graph_id, :integer
      t.column :datasource_id, :integer
      t.column :negative, :boolean
      t.column :sortorder, :integer
    end
  end

  def self.down
    drop_table :graph_datasources
  end
end
