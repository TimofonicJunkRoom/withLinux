push!(LOAD_PATH, pwd())

import lumin_log
log = lumin_log

@log.info("info")
@log.debug("debug")
@log.warn("warn")
@log.error("error")
@log.fatal("fatal")

include("lumin_log.jl")
@lumin_log.info("asdf")
