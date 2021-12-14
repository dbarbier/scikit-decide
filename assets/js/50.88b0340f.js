(window.webpackJsonp=window.webpackJsonp||[]).push([[50],{563:function(t,e,a){"use strict";a.r(e);var i=a(38),s=Object(i.a)({},(function(){var t=this,e=t.$createElement,a=t._self._c||e;return a("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[a("h1",{attrs:{id:"builders-domain-scheduling-time-lag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#builders-domain-scheduling-time-lag"}},[t._v("#")]),t._v(" builders.domain.scheduling.time_lag")]),t._v(" "),a("div",{staticClass:"custom-block tip"},[a("p",{staticClass:"custom-block-title"},[t._v("Domain specification")]),t._v(" "),a("skdecide-summary")],1),t._v(" "),a("h2",{attrs:{id:"timelag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#timelag"}},[t._v("#")]),t._v(" TimeLag")]),t._v(" "),a("p",[t._v("Defines a time lag with both a minimum time lag and maximum time lag.")]),t._v(" "),a("h3",{attrs:{id:"constructor"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor"}},[t._v("#")]),t._v(" Constructor "),a("Badge",{attrs:{text:"TimeLag",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"TimeLag",sig:{params:[{name:"minimum_time_lag"},{name:"maximum_time_lags"}]}}}),t._v(" "),a("p",[t._v("Initialize self.  See help(type(self)) for accurate signature.")]),t._v(" "),a("h2",{attrs:{id:"minimumonlytimelag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#minimumonlytimelag"}},[t._v("#")]),t._v(" MinimumOnlyTimeLag")]),t._v(" "),a("p",[t._v("Defines a minimum time lag.")]),t._v(" "),a("h3",{attrs:{id:"constructor-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor-2"}},[t._v("#")]),t._v(" Constructor "),a("Badge",{attrs:{text:"MinimumOnlyTimeLag",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"MinimumOnlyTimeLag",sig:{params:[{name:"minimum_time_lag"}]}}}),t._v(" "),a("p",[t._v("Initialize self.  See help(type(self)) for accurate signature.")]),t._v(" "),a("h2",{attrs:{id:"maximumonlytimelag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#maximumonlytimelag"}},[t._v("#")]),t._v(" MaximumOnlyTimeLag")]),t._v(" "),a("p",[t._v("Defines a maximum time lag.")]),t._v(" "),a("h3",{attrs:{id:"constructor-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#constructor-3"}},[t._v("#")]),t._v(" Constructor "),a("Badge",{attrs:{text:"MaximumOnlyTimeLag",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"MaximumOnlyTimeLag",sig:{params:[{name:"maximum_time_lags"}]}}}),t._v(" "),a("p",[t._v("Initialize self.  See help(type(self)) for accurate signature.")]),t._v(" "),a("h2",{attrs:{id:"withtimelag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#withtimelag"}},[t._v("#")]),t._v(" WithTimeLag")]),t._v(" "),a("p",[t._v("A domain must inherit this class if there are minimum and maximum time lags between some of its tasks.")]),t._v(" "),a("h3",{attrs:{id:"get-time-lags"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-time-lags"}},[t._v("#")]),t._v(" get_time_lags "),a("Badge",{attrs:{text:"WithTimeLag",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_time_lags",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, TimeLag]]"}}}),t._v(" "),a("p",[t._v("Return nested dictionaries where the first key is the id of a task (int)\nand the second key is the id of another task (int).\nThe value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end\nof the first task to the start of the second task.")]),t._v(" "),a("p",[t._v("e.g.\n{\n12:{\n15: TimeLag(5, 10),\n16: TimeLag(5, 20),\n17: MinimumOnlyTimeLag(5),\n18: MaximumOnlyTimeLag(15),\n}\n}")]),t._v(" "),a("h4",{attrs:{id:"returns"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("A dictionary of TimeLag objects.")]),t._v(" "),a("h3",{attrs:{id:"get-time-lags-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-time-lags-2"}},[t._v("#")]),t._v(" _get_time_lags "),a("Badge",{attrs:{text:"WithTimeLag",type:"tip"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_time_lags",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, TimeLag]]"}}}),t._v(" "),a("p",[t._v("Return nested dictionaries where the first key is the id of a task (int)\nand the second key is the id of another task (int).\nThe value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end\nof the first task to the start of the second task.")]),t._v(" "),a("p",[t._v("e.g.\n{\n12:{\n15: TimeLag(5, 10),\n16: TimeLag(5, 20),\n17: MinimumOnlyTimeLag(5),\n18: MaximumOnlyTimeLag(15),\n}\n}")]),t._v(" "),a("h4",{attrs:{id:"returns-2"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-2"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("A dictionary of TimeLag objects.")]),t._v(" "),a("h2",{attrs:{id:"withouttimelag"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#withouttimelag"}},[t._v("#")]),t._v(" WithoutTimeLag")]),t._v(" "),a("p",[t._v("A domain must inherit this class if there is no required time lag between its tasks.")]),t._v(" "),a("h3",{attrs:{id:"get-time-lags-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-time-lags-3"}},[t._v("#")]),t._v(" get_time_lags "),a("Badge",{attrs:{text:"WithTimeLag",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"get_time_lags",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, TimeLag]]"}}}),t._v(" "),a("p",[t._v("Return nested dictionaries where the first key is the id of a task (int)\nand the second key is the id of another task (int).\nThe value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end\nof the first task to the start of the second task.")]),t._v(" "),a("p",[t._v("e.g.\n{\n12:{\n15: TimeLag(5, 10),\n16: TimeLag(5, 20),\n17: MinimumOnlyTimeLag(5),\n18: MaximumOnlyTimeLag(15),\n}\n}")]),t._v(" "),a("h4",{attrs:{id:"returns-3"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#returns-3"}},[t._v("#")]),t._v(" Returns")]),t._v(" "),a("p",[t._v("A dictionary of TimeLag objects.")]),t._v(" "),a("h3",{attrs:{id:"get-time-lags-4"}},[a("a",{staticClass:"header-anchor",attrs:{href:"#get-time-lags-4"}},[t._v("#")]),t._v(" _get_time_lags "),a("Badge",{attrs:{text:"WithTimeLag",type:"warn"}})],1),t._v(" "),a("skdecide-signature",{attrs:{name:"_get_time_lags",sig:{params:[{name:"self"}],return:"Dict[int, Dict[int, TimeLag]]"}}}),t._v(" "),a("p",[t._v("Return nested dictionaries where the first key is the id of a task (int)\nand the second key is the id of another task (int).\nThe value is a TimeLag object containing the MINIMUM and MAXIMUM time (int) that needs to separate the end\nof the first task to the start of the second task.")])],1)}),[],!1,null,null,null);e.default=s.exports}}]);