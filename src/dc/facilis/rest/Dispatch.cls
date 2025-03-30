Class dc.facilis.rest.Dispatch Extends %CSP.REST
{

Parameter HandleCorsRequest = "true";

/// By default convert the input stream to Unicode
Parameter CONVERTINPUTSTREAM = 1;

/// The default response charset is utf-8
Parameter CHARSET = "utf-8";

XData UrlMap [ XMLNamespace = "http://www.intersystems.com/urlmap" ]
{
<Routes>
    <Route Url="namespaces" Method="get" Call="NamespacesList" />
    <!-- Generate Interoperability from OpenAPI -->
    <Route Url="generate" Method="post" Call="Generate" />
</Routes>
}

ClassMethod NamespacesList() As %Status
{
    Try {
        Do ##class(%REST.Impl).%SetContentType("application/json")
        Set response = []

        Do ##class(%SYS.Namespace).ListAll(.result)
        Set ns = ""
        While ($Order(result(ns))'="") {
            Set ns = $Order(result(ns))
            Continue:(ns = "%SYS")
            Do response.%Push(ns)
        }
        Do ##class(dc.facilis.rest.Impl).%WriteResponse(response)
    } Catch (ex) {
        Do ##class(dc.facilis.rest.Impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("dc.facilis.rest.Impl","ExposeServerExceptions"))
    }
    Return $$$OK
}

/// Generate Interoperability
ClassMethod Generate() As %Status
{
    Try {
        If $case(%request.ContentType,"application/json":0,:1) Do ##class(dc.facilis.rest.Impl).%ReportRESTError(..#HTTP415UNSUPPORTEDMEDIATYPE,$$$ERROR($$$RESTContentType,%request.ContentType)) Quit
        Do ##class(%REST.Impl).%SetContentType("application/json")
        If '##class(%REST.Impl).%CheckAccepts("application/json") Do ##class(dc.facilis.rest.Impl).%ReportRESTError(..#HTTP406NOTACCEPTABLE,$$$ERROR($$$RESTBadAccepts)) Quit
        If '$isobject(%request.Content) || (%request.Content.Size=0) Do ##class(dc.facilis.rest.Impl).%ReportRESTError(..#HTTP400BADREQUEST,$$$ERROR($$$RESTRequired,"input")) Quit
        Set pinput=##class(%DynamicAbstractObject).%FromJSON(%request.Content)

        $$$ThrowOnError(##class(dc.facilis.core).Execute(pinput, .response))
        Do ##class(dc.facilis.rest.Impl).%WriteResponse(response)
    } Catch (ex) {
        Do ##class(dc.facilis.rest.Impl).%ReportRESTError(..#HTTP500INTERNALSERVERERROR,ex.AsStatus(),$parameter("dc.facilis.rest.Impl","ExposeServerExceptions"))
    }
}

}
