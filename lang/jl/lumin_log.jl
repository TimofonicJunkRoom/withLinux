# GLOG style, simple julia logging helper
module lumin_log

#@printf("this provides some debug information\n")
#println("filename ", @__FILE__, " line number ", @__LINE__)
#
## this comes from https://github.com/JuliaLang/julia/issues/8066
#macro __LINENO__()
#	return :(unsafe_load(Ptr{Int}(cglobal(:jl_lineno))))
#end
#println("line bumber ", @__LINENO__())

# FIXME: __LINE__ and __FILE__ do not work as expected.
# https://github.com/JuliaLang/julia/issues/8066
# https://github.com/c42f/MicroLogging.jl/issues/1
# https://github.com/JuliaLang/julia/issues/9577

macro _lumin_log(msg)
	return esc(:(@sprintf("%s %d %s:%d] %s",
		 Dates.format(now(), "mmdd H:M:S"), getpid(), basename(@__FILE__),
		 @__LINE__, msg )))
end

macro debug(msg)
	println("\x1b[1;36mD", @_lumin_log(msg), "\x1b[0;m")
end

macro info(msg)
	println("\x1b[1;32mI", @_lumin_log(msg), "\x1b[0;m")
end

macro warn(msg)
	println("\x1b[1;33mW", @_lumin_log(msg), "\x1b[0;m")
end

macro error(msg)
	println("\x1b[1;31mE", @_lumin_log(msg), "\x1b[0;m")
end

macro fatal(msg)
	println("\x1b[1;35mF", @_lumin_log(msg), "\x1b[0;m")
	if VERSION > v"0.5.0" # v"0.4.7" doesn't support the following stuff
		stacktrace(backtrace())
	end
end

export @debug, @info, @warn, @error, @fatal

@debug("debug test")
@info("info test")
@warn("warning test")
@error("error test")
@fatal("fatal test")

end # module lumin_log
