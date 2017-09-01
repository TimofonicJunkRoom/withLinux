#!/usr/bin/th
-- @ref https://github.com/Yonaba/Moses/blob/master/doc/tutorial.md
local _ = require "moses"

-- table functions ____________________________________________________________

_.each({1,2,3}, print)
_.eachi({1,2,3}, print)
print(_.at({4,5,6}, 1,2)) -- 4,5
print(_.count({1,1,1,1,2,3,4}, 1)) -- 4
print(_.countf({1,2,3,4,5,6}, function(i,v) return v%2==0 end)) -- 3
print(_.map({1,2,3,4}, function(i,v) return v*v end))
print(_.reduce({1,2,3,4}, function(memo,v) return memo+v end))
print(_.detect({1,2,3,4}, false)) -- nil
print(_.select({1,2,3,4,5,6}, function(k,v) return v%2==0 end))
print(_.reject({1,2,3,4,5,6}, function(k,v) return v%2==0 end))
print(_.all({1,2,3}, function(_,v) return v>0 end))
print(_.max({1,2,3}))
print(_.min({1,2,3}))
print(_.shuffle({1,2,3}))

-- array functions ____________________________________________________________

print(_.sample(_.range(100,254), 3))


-- utility functions __________________________________________________________




-- object functions ___________________________________________________________






