Class dc.facilis.rest.Impl Extends %REST.Impl [ ProcedureBlock ]
{

/// If ExposeServerExceptions is true, then details of internal errors will be exposed.
Parameter ExposeServerExceptions = 0;

ClassMethod %ReportRESTError(statusCode As %String, sc As %Status, exposeServerExceptions As %Boolean = 0)
{
    Set ex = ##class(%Exception.StatusException).CreateFromStatus(sc)

    Set errorDesc = ex.DisplayString()

    JOB ..Log(ex.DisplayString())
    Do ##super(statusCode, sc, exposeServerExceptions)
}

/// Copied from %Exception.AbstractException
/// Call the log function (LOG^%ETN) to log this exception.
/// You can view this log at the terminal with 'Do ^%ER' utility or from the system management portal.
ClassMethod Log(msg) As %Status
{
	try {
		do LOG^%ETN(msg)
		set sc=$$$OK
	} catch err {
		set sc=err.AsStatus()
	}
	quit sc
}

}