pipeline {
	agent { node 'agent-coe' }

	environment {
	    LC_ALL = 'C.UTF-8'
        LANG   = 'C.UTF-8'
	}

	stages {
	  	stage('Setup') {
			steps {
				sh '''
				  make init
				'''
			}
		}

	  	stage('Lint') {
			steps {
				sh '''
				make lint
				'''
			}
		}

	  	stage('Unit Tests') {
			steps {
				sh '''
				make unittest
				'''
			}
		}
	}

	post {
		always {
			junit "test_report.xml"
			cobertura autoUpdateHealth: false, autoUpdateStability: false, coberturaReportFile: 'coverage.xml', conditionalCoverageTargets: '70, 0, 0', failUnhealthy: false, failUnstable: false, lineCoverageTargets: '80, 0, 0', maxNumberOfBuilds: 0, methodCoverageTargets: '80, 0, 0', onlyStable: false, sourceEncoding: 'ASCII', zoomCoverageChart: false
		}
	}
}