#include <lua5.1/lua.h>
#include <TH/TH.h>
#include <luaT.h>

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define thnnauxmod_(NAME) TH_CONCAT_3(thnnauxmod_, Real, NAME)

#include "mythreshold.c"
#include <TH/THGenerateFloatTypes.h>

LUA_EXTERNC DLL_EXPORT int luaopen_libthnnauxmod(lua_State *L);

int luaopen_libthnnauxmod(lua_State *L)
{
	lua_newtable(L);
	lua_pushvalue(L, -1);
	lua_setglobal(L, "thnnauxmod");

	thnnauxmod_FloatMyThreshold_init(L);
	thnnauxmod_DoubleMyThreshold_init(L);

	return 1;
}
