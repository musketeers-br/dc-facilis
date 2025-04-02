Class dc.facilis.Log Extends %RegisteredObject
{

ClassMethod Log(level, location, msg) As %Status
{
    Set sc = $$$OK
    Try {
        TStart
        Set ^facilis.log($Translate($ZDateTime($Horolog, 8)," :","_"), level, location) = msg
        TCommit
    } Catch tException {
        TRollback
        Set:$$$ISOK(sc) sc = tException.AsStatus()
    }
    Return sc
}

ClassMethod Info(location, msg = "") As %Status
{
    Return ..Log("INFO", location, msg)
}

ClassMethod Error(location, msg = "") As %Status
{
    Return ..Log("ERROR", location, msg)
}

ClassMethod Warning(location, msg = "") As %Status
{
    Return ..Log("WARNING", location, msg)
}

}
