class CreateDatasources < ActiveRecord::Migration
  def self.up
    create_table :datasources do |t|
			t.column :name, :string
			t.column :rrd, :string
			t.column :datatype, :string
			t.column :parent_id, :integer
    end
  end

  def self.down
    drop_table :datasources
  end
end
