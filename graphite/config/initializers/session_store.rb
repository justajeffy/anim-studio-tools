# Be sure to restart your server when you modify this file.

# Your secret key for verifying cookie session data integrity.
# If you change this key, all old sessions will become invalid!
# Make sure the secret is at least 30 characters and all random, 
# no regular words or you'll be exposed to dictionary attacks.
ActionController::Base.session = {
  :key         => '_noc_session',
  :secret      => '4917dbf5b444e8e068c19105f7545212d26c8e241720ad4390c33e4b4d4e74683c22fae85fb4433acc97456b99c52c2c4476c911ab6f987d3c8c5738aefdcf66'
}

# Use the database for sessions instead of the cookie-based default,
# which shouldn't be used to store highly confidential information
# (create the session table with "rake db:sessions:create")
# ActionController::Base.session_store = :active_record_store
